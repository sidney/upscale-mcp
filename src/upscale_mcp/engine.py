"""Model loading, tiled inference and image I/O for upscale-mcp.

Nothing in this module knows about MCP, so it can be exercised from a plain
Python prompt or from the tests. server.py is the thin MCP layer on top.

Pipeline for one call:

    load_image()  ->  upscale_array()  ->  optional LANCZOS resize  ->  save_image()

upscale_array() runs the model tile by tile. Each tile is cut from the source
with an extra `overlap` margin of context on every side, run through the model,
and only the scaled-up *core* of the tile (without the margin) is written into
the output. Because every output pixel then comes from a tile that saw at
least `overlap` source pixels of context around it, seams disappear as long as
the overlap covers the model's effective receptive field near the edges.
"""

from __future__ import annotations

import os

# Must be set before torch is imported. Some operators used by upscaling
# architectures have no Metal (MPS) kernel; with this set, PyTorch runs just
# those operators on the CPU instead of raising NotImplementedError. It costs
# speed only for the affected operators.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Protocol  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from spandrel import ImageModelDescriptor, ModelLoader  # noqa: E402

# File types spandrel can load. .pth/.pt/.ckpt are pickled PyTorch checkpoints;
# spandrel loads them with torch's restricted ("weights_only") unpickler, so a
# malicious checkpoint cannot run arbitrary code the way plain torch.load could.
# .safetensors is a plain tensor container and safe by construction.
MODEL_SUFFIXES = (".pth", ".pt", ".safetensors", ".ckpt")

# Output formats we know how to write, keyed by lower-case suffix.
OUTPUT_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")

ProgressFn = Callable[[int, int], None]


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


def pick_device(preference: str = "auto") -> torch.device:
    """Return the torch device to run models on.

    "auto" prefers Apple's Metal backend (mps), then CUDA, then the CPU. An
    explicit preference that is not available raises instead of silently
    falling back, so a misconfigured server fails loudly at startup.
    """
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(preference)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but torch.backends.mps is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return device


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """What spandrel tells us about a model file, in JSON-friendly form."""

    name: str  # file name, e.g. "RealESRGAN_x4plus.pth"
    path: str
    size_mb: float
    architecture: str  # e.g. "RealESRGAN" / "ESRGAN", "SwinIR", "HAT", "DAT"
    purpose: str  # spandrel's purpose tag: "SR", "Restoration", "FaceSR", ...
    scale: int  # 1 for restoration models, 2/4/8 for upscalers
    input_channels: int
    output_channels: int
    supports_half: bool
    tags: tuple[str, ...]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def find_model_files(models_dirs: list[Path]) -> list[Path]:
    """All model files under the configured directories (recursively).

    Recursive so that a directory like ComfyUI's models/upscale_models, which
    people often organise into subfolders, works as-is. Hidden files and
    folders (dot-names) are skipped.
    """
    found: list[Path] = []
    for root in models_dirs:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES:
                found.append(p)
    return found


def resolve_model(name: str, models_dirs: list[Path]) -> Path:
    """Map a model name from a tool call to a file.

    Accepts the exact file name ("RealESRGAN_x4plus.pth"), the stem without
    extension ("RealESRGAN_x4plus"), or a path relative to a models dir
    ("photo/4x_foo.safetensors"). Matching is case-insensitive. Anything that
    looks like an absolute path or escapes the models dirs is refused, so a
    tool call can only ever load files you put in the models folders.
    """
    files = find_model_files(models_dirs)
    wanted = name.strip().lower()
    if not wanted or Path(wanted).is_absolute() or ".." in Path(wanted).parts:
        raise ValueError(f"invalid model name {name!r}: give a file name from list_models")

    def rel(p: Path) -> str:
        for root in models_dirs:
            try:
                return p.relative_to(root).as_posix().lower()
            except ValueError:
                continue
        return p.name.lower()

    for key in (
        lambda p: rel(p),  # "photo/4x_foo.safetensors"
        lambda p: p.name.lower(),  # "4x_foo.safetensors"
        lambda p: p.stem.lower(),  # "4x_foo"
    ):
        matches = [p for p in files if key(p) == wanted]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            listed = ", ".join(rel(p) for p in matches)
            raise ValueError(f"model name {name!r} is ambiguous: {listed}")
    available = ", ".join(p.name for p in files) or "(none)"
    raise FileNotFoundError(f"no model named {name!r} in the models folders; available: {available}")


def _describe(desc: ImageModelDescriptor, path: Path) -> ModelInfo:
    return ModelInfo(
        name=path.name,
        path=str(path),
        size_mb=round(path.stat().st_size / 1e6, 1),
        architecture=desc.architecture.name,
        purpose=str(desc.purpose),
        scale=int(desc.scale),
        input_channels=int(desc.input_channels),
        output_channels=int(desc.output_channels),
        supports_half=bool(desc.supports_half),
        tags=tuple(desc.tags),
    )


def _load_descriptor(path: Path) -> ImageModelDescriptor:
    desc = ModelLoader().load_from_file(path)  # always loads onto the CPU
    if not isinstance(desc, ImageModelDescriptor):
        # e.g. inpainting models (MaskedImageModelDescriptor) need a mask input.
        raise ValueError(f"{path.name} is a {type(desc).__name__}, not a single-image model")
    return desc


class ModelCache:
    """Keeps the most recently used model resident on the device.

    Loading a model and moving it to the GPU takes a second or two; keeping
    the last one around makes repeated calls with the same model (the usual
    case while comparing settings) fast. Only one model is kept on the device
    to bound GPU memory. Metadata for list_models is cached separately per
    (path, mtime) so listing does not keep reloading every file.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._desc: ImageModelDescriptor | None = None
        self._info_cache: dict[tuple[str, float], ModelInfo] = {}

    def info(self, path: Path) -> ModelInfo:
        key = (str(path), path.stat().st_mtime)
        with self._lock:
            cached = self._info_cache.get(key)
        if cached is not None:
            return cached
        info = _describe(_load_descriptor(path), path)
        with self._lock:
            self._info_cache[key] = info
        return info

    def get(self, path: Path, device: torch.device, half: bool) -> tuple[ImageModelDescriptor, ModelInfo]:
        key = (str(path), path.stat().st_mtime, str(device), half)
        with self._lock:
            if self._key == key and self._desc is not None:
                return self._desc, self._info_cache[(str(path), key[1])]
            # Drop the previous model before loading the next one.
            self._key = None
            self._desc = None
        if device.type == "mps":
            torch.mps.empty_cache()
        desc = _load_descriptor(path)
        info = _describe(desc, path)
        if half and not desc.supports_half:
            raise ValueError(f"{path.name} ({info.architecture}) does not support half precision")
        if half and device.type == "cpu":
            raise ValueError("half precision is not useful on the CPU; leave half=false")
        desc.to(device)
        if half:
            desc.half()
        desc.eval()
        with self._lock:
            self._key = key
            self._desc = desc
            self._info_cache[(str(path), key[1])] = info
        return desc, info


# --------------------------------------------------------------------------
# Tiled inference
# --------------------------------------------------------------------------


class SizeReq(Protocol):
    """The parts of spandrel's SizeRequirements we use (duck-typed for tests)."""

    minimum: int
    multiple_of: int
    square: bool


def _pad_for_model(t: torch.Tensor, req: SizeReq) -> torch.Tensor:
    """Pad a BCHW tile on the bottom/right so the model accepts its size.

    Some architectures (transformer models such as SwinIR, HAT, DAT) need the
    input to be a multiple of their window size, a minimum size, or square.
    "replicate" padding repeats the edge pixels; the padded area is cropped
    off the output again, so its content only matters as context for the
    real pixels next to it, where repeating the edge is the least surprising.
    """
    _, _, h, w = t.shape
    th, tw = h, w
    if req.square:
        th = tw = max(th, tw)
    th = max(th, req.minimum)
    tw = max(tw, req.minimum)
    m = max(1, req.multiple_of)
    th = math.ceil(th / m) * m
    tw = math.ceil(tw / m) * m
    if (th, tw) == (h, w):
        return t
    return F.pad(t, (0, tw - w, 0, th - h), mode="replicate")


def tile_grid(height: int, width: int, tile: int) -> list[tuple[int, int, int, int]]:
    """Core tile rectangles (y0, x0, y1, x1) covering the image.

    tile <= 0, or a tile at least as large as the image, means one tile.
    """
    if tile <= 0 or tile >= max(height, width):
        return [(0, 0, height, width)]
    return [
        (y0, x0, min(y0 + tile, height), min(x0 + tile, width))
        for y0 in range(0, height, tile)
        for x0 in range(0, width, tile)
    ]


def upscale_array(
    rgb: np.ndarray,
    model: Callable[[torch.Tensor], torch.Tensor],
    scale: int,
    size_req: SizeReq,
    device: torch.device,
    *,
    half: bool = False,
    tile: int = 512,
    overlap: int = 32,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Run `model` over an H x W x 3 uint8 image; return (H*scale) x (W*scale) x 3 uint8.

    `model` takes and returns BCHW float tensors in [0, 1] on `device`
    (spandrel's ImageModelDescriptor is such a callable). The source stays on
    the CPU as uint8; only one context-padded tile at a time is moved to the
    device, which keeps GPU memory proportional to the tile size, not the
    image size. The output is assembled on the CPU as uint8.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected H x W x 3 uint8, got {rgb.shape} {rgb.dtype}")
    height, width = rgb.shape[:2]
    out = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    # torch.from_numpy shares memory and warns on read-only arrays (which is
    # what np.asarray(PIL image) returns), so copy those; the source is never
    # written to, but a copy keeps the log clean and costs only H*W*3 bytes.
    rgb = np.ascontiguousarray(rgb)
    if not rgb.flags.writeable:
        rgb = rgb.copy()
    src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW uint8
    dtype = torch.float16 if half else torch.float32
    tiles = tile_grid(height, width, tile)

    with torch.inference_mode():
        for i, (y0, x0, y1, x1) in enumerate(tiles):
            # Context rectangle: the core plus `overlap` pixels each side,
            # clamped to the image.
            cy0, cx0 = max(0, y0 - overlap), max(0, x0 - overlap)
            cy1, cx1 = min(height, y1 + overlap), min(width, x1 + overlap)
            patch = src[:, :, cy0:cy1, cx0:cx1].to(device=device, dtype=dtype) / 255.0
            ph, pw = patch.shape[2], patch.shape[3]
            result = model(_pad_for_model(patch, size_req))
            # Drop any model-size padding, then keep only the scaled core.
            result = result[:, :, : ph * scale, : pw * scale]
            result = result[
                :,
                :,
                (y0 - cy0) * scale : (y1 - cy0) * scale,
                (x0 - cx0) * scale : (x1 - cx0) * scale,
            ]
            # Round in float32 regardless of compute precision.
            block = (result.float().clamp_(0, 1) * 255.0).round_().to(torch.uint8)
            out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = block[0].permute(1, 2, 0).cpu().numpy()
            if progress is not None:
                progress(i + 1, len(tiles))
    return out


# --------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------


@dataclass
class LoadedImage:
    rgb: np.ndarray  # H x W x 3 uint8
    alpha: np.ndarray | None  # H x W uint8, or None when the image is opaque
    icc_profile: bytes | None  # embedded colour profile, carried to the output
    source_mode: str  # PIL mode of the file, for reporting

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]  # (width, height), PIL order


def load_image(path: Path) -> LoadedImage:
    """Read an image as 8-bit RGB plus optional alpha.

    The EXIF orientation is applied to the pixels (and so dropped from the
    output). Apps that honour the orientation tag, like PhotoCraft, show the
    original upright; baking the rotation in makes the output line up with
    what they show instead of coming out sideways.

    The ICC profile is kept so colours do not shift when the output is
    placed back over the original. A CMYK source's profile is dropped
    because the pixels have been converted to RGB.
    """
    with Image.open(path) as im:
        im.load()
        source_mode = im.mode
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        if source_mode == "CMYK":
            icc = None
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        alpha = np.asarray(im.convert("RGBA").getchannel("A")).copy() if has_alpha else None
        rgb = np.asarray(im.convert("RGB")).copy()
    return LoadedImage(rgb=rgb, alpha=alpha, icc_profile=icc, source_mode=source_mode)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """LANCZOS resize to (width, height).

    PIL's LANCZOS widens its kernel when shrinking, so a 4x reduction is
    properly anti-aliased rather than just sampled.
    """
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb
    return np.asarray(Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.LANCZOS))


def resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (alpha.shape[1], alpha.shape[0]) == size:
        return alpha
    return np.asarray(Image.fromarray(alpha, "L").resize(size, Image.Resampling.LANCZOS))


def save_image(
    path: Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    icc_profile: bytes | None,
    *,
    jpeg_quality: int = 95,
) -> list[str]:
    """Write the result; the format comes from the file suffix.

    Returns warnings (for example alpha dropped for JPEG) for the caller.
    """
    warnings: list[str] = []
    suffix = path.suffix.lower()
    if suffix not in OUTPUT_SUFFIXES:
        raise ValueError(f"unsupported output type {suffix!r}; use one of {', '.join(OUTPUT_SUFFIXES)}")
    im = Image.fromarray(rgb, "RGB")
    if alpha is not None:
        if suffix in (".jpg", ".jpeg"):
            warnings.append("JPEG has no alpha channel; transparency was dropped")
        else:
            im.putalpha(Image.fromarray(alpha, "L"))
    kwargs: dict = {}
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    if suffix == ".png":
        kwargs["compress_level"] = 6
    elif suffix in (".jpg", ".jpeg"):
        # 4:4:4 chroma: the point of the exercise is fine detail, so do not
        # throw half the colour resolution away on save.
        kwargs.update(quality=jpeg_quality, subsampling=0)
    elif suffix in (".tif", ".tiff"):
        kwargs["compression"] = "tiff_lzw"
    elif suffix == ".webp":
        kwargs["lossless"] = True
    im.save(path, **kwargs)
    return warnings


# --------------------------------------------------------------------------
# One complete job
# --------------------------------------------------------------------------


@dataclass
class UpscaleResult:
    input: str
    output: str
    input_size: tuple[int, int]  # (width, height)
    model_output_size: tuple[int, int]
    output_size: tuple[int, int]
    model: dict
    device: str
    half: bool
    tiles: int
    tile_size: int
    tile_overlap: int
    seconds: float
    warnings: list[str]

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("input_size", "model_output_size", "output_size"):
            d[k] = list(d[k])
        return d


def upscale_file(
    src: Path,
    dst: Path,
    model_path: Path,
    cache: ModelCache,
    device: torch.device,
    *,
    output_scale: float | None = None,
    tile: int = 512,
    overlap: int = 32,
    half: bool = False,
    jpeg_quality: int = 95,
    progress: ProgressFn | None = None,
) -> UpscaleResult:
    """Upscale `src` with `model_path` and write `dst`.

    output_scale is relative to the *input* size:
      None -> keep the model's native scale (a 4x model gives 4x);
      1.0  -> resize back to the input size (the "upscale then downscale"
              trick: the model's added detail survives as crisper texture,
              and the result registers pixel for pixel with the original);
      any other value -> that multiple of the input size.
    """
    started = time.perf_counter()
    desc, info = cache.get(model_path, device, half)
    if info.input_channels != 3 or info.output_channels != 3:
        raise ValueError(
            f"{info.name} works on {info.input_channels}-channel input and "
            f"{info.output_channels}-channel output; only RGB models are supported"
        )
    image = load_image(src)
    width, height = image.size
    upscaled = upscale_array(
        image.rgb,
        desc,
        info.scale,
        desc.size_requirements,
        device,
        half=half,
        tile=tile,
        overlap=overlap,
        progress=progress,
    )
    model_size = (upscaled.shape[1], upscaled.shape[0])
    if output_scale is None:
        final_size = model_size
    else:
        final_size = (max(1, round(width * output_scale)), max(1, round(height * output_scale)))
    final_rgb = resize_rgb(upscaled, final_size)
    final_alpha = resize_alpha(image.alpha, final_size) if image.alpha is not None else None
    warnings = save_image(dst, final_rgb, final_alpha, image.icc_profile, jpeg_quality=jpeg_quality)
    if image.alpha is not None:
        warnings.append("alpha channel was resized with LANCZOS, not run through the model")
    return UpscaleResult(
        input=str(src),
        output=str(dst),
        input_size=(width, height),
        model_output_size=model_size,
        output_size=final_size,
        model=info.as_dict(),
        device=str(device),
        half=half,
        tiles=len(tile_grid(height, width, tile)),
        tile_size=tile,
        tile_overlap=overlap,
        seconds=round(time.perf_counter() - started, 2),
        warnings=warnings,
    )
