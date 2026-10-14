"""Model loading, tiled inference and image I/O for upscale-mcp.

Nothing in this module knows about MCP, so it can be exercised from a plain
Python prompt or from the tests. server.py is the thin MCP layer on top.

Pipeline for one call:

    load_image()  ->  upscale_array()  ->  optional LANCZOS resize  ->  save_image()

Images are 8-bit or 16-bit end to end, following the source (see the Image
I/O section for which formats carry more than 8 bits and how).

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

# Output formats we know how to write: lower-case suffix -> PIL format name.
# The format is passed to PIL explicitly because the file is first written
# under a temporary name (see save_image), whose suffix PIL would not know.
OUTPUT_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
OUTPUT_SUFFIXES = tuple(OUTPUT_FORMATS)

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
    """Run `model` over an H x W x 3 image; return (H*scale) x (W*scale) x 3 of the same dtype.

    The image may be uint8 (0..255) or uint16 (full 0..65535 range, see
    load_image). Models work in float [0, 1] either way, so a 16-bit source
    simply reaches the model with finer steps and the result keeps them.

    `model` takes and returns BCHW float tensors in [0, 1] on `device`
    (spandrel's ImageModelDescriptor is such a callable). The source stays on
    the CPU in its integer form; each context-padded tile is converted to
    float32 in numpy and moved to the device on its own, which keeps GPU
    memory proportional to the tile size, not the image size. (Converting in
    numpy also avoids torch's limited uint16 support.) The output is
    assembled on the CPU in the source dtype.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"expected H x W x 3 uint8 or uint16, got {rgb.shape} {rgb.dtype}")
    maxval = float(np.iinfo(rgb.dtype).max)
    height, width = rgb.shape[:2]
    out = np.empty((height * scale, width * scale, 3), dtype=rgb.dtype)
    dtype = torch.float16 if half else torch.float32
    tiles = tile_grid(height, width, tile)

    with torch.inference_mode():
        for i, (y0, x0, y1, x1) in enumerate(tiles):
            # Context rectangle: the core plus `overlap` pixels each side,
            # clamped to the image.
            cy0, cx0 = max(0, y0 - overlap), max(0, x0 - overlap)
            cy1, cx1 = min(height, y1 + overlap), min(width, x1 + overlap)
            chw = rgb[cy0:cy1, cx0:cx1].transpose(2, 0, 1).astype(np.float32) / maxval
            patch = torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=dtype)
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
            # Round in float32 regardless of compute precision; go through
            # int32 because torch's uint16 support is limited.
            block = (result.float().clamp_(0, 1) * maxval).round_().to(torch.int32)
            out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = (
                block[0].permute(1, 2, 0).cpu().numpy().astype(rgb.dtype)
            )
            if progress is not None:
                progress(i + 1, len(tiles))
    return out


# --------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------
#
# Images travel through the pipeline as uint8 (8-bit sources) or uint16 using
# the full 0..65535 range (anything deeper than 8 bits: 10/12-bit HEIF,
# 16-bit PNG and TIFF). The output keeps the source's depth where the output
# format can hold it (PNG, TIFF) and is rounded to 8 bits otherwise (JPEG,
# WebP), with a warning.
#
# Readers:
#   .heic/.heif/.avif   pillow-heif, at the file's own bit depth
#   >8-bit .png/.tif    OpenCV for the pixels, because Pillow silently
#                       truncates 48/64-bit RGB PNGs and TIFFs to 8 bits;
#                       Pillow still supplies the metadata (ICC profile,
#                       EXIF orientation)
#   everything else     Pillow, 8 bits, as before
# Writers:
#   8-bit               Pillow, as before
#   16-bit .png         OpenCV, with the ICC profile added as an iCCP chunk
#                       (OpenCV cannot write one itself)
#   16-bit .tif         tifffile, with the ICC profile in its standard tag

HEIF_SUFFIXES = (".heic", ".heif", ".avif")


@dataclass
class LoadedImage:
    rgb: np.ndarray  # H x W x 3, uint8, or uint16 over the full 0..65535 range
    alpha: np.ndarray | None  # H x W, same dtype as rgb, or None when opaque
    icc_profile: bytes | None  # embedded colour profile, carried to the output
    source_mode: str  # how the file was encoded, for reporting
    source_bits: int  # bits per channel in the file: 8, 10, 12 or 16
    warnings: list[str]

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]  # (width, height), PIL order


def expand_to_16(a: np.ndarray, bits: int) -> np.ndarray:
    """Rescale `bits`-deep codes stored left-aligned in uint16 to the full 0..65535 range.

    pillow-heif returns 10-bit data shifted into the top of 16 bits, so white
    is 1023 << 6 = 65472, not 65535. Dividing that by 65535 would make every
    pixel about 0.1% too dark. Shifting back down to the real codes and
    rescaling so the brightest code becomes 65535 keeps white white.
    """
    if bits >= 16:
        return a
    codes = (a >> (16 - bits)).astype(np.uint32)
    maxcode = (1 << bits) - 1
    return ((codes * 65535 + maxcode // 2) // maxcode).astype(np.uint16)


def to_8bit(a: np.ndarray) -> np.ndarray:
    """Round full-range 16-bit values to 8 bits (65535 / 257 = 255 exactly)."""
    if a.dtype == np.uint8:
        return a
    return ((a.astype(np.uint32) + 128) // 257).astype(np.uint8)


def apply_orientation(a: np.ndarray, orientation: int) -> np.ndarray:
    """Turn pixels upright according to an EXIF orientation value (1..8).

    Matches PIL.ImageOps.exif_transpose, which handles the 8-bit path; this
    is for arrays Pillow cannot hold (16-bit RGB).
    """
    if orientation == 2:
        a = a[:, ::-1]
    elif orientation == 3:
        a = np.rot90(a, 2)
    elif orientation == 4:
        a = a[::-1]
    elif orientation == 5:
        a = a.swapaxes(0, 1)
    elif orientation == 6:
        a = np.rot90(a, -1)
    elif orientation == 7:
        a = np.rot90(a, 2).swapaxes(0, 1)
    elif orientation == 8:
        a = np.rot90(a, 1)
    return np.ascontiguousarray(a)


def _load_heif(path: Path) -> LoadedImage:
    """HEIC / HEIF / AVIF through pillow-heif, keeping 10- and 12-bit data.

    libheif applies the container's own rotation and mirroring (irot/imir)
    while decoding; the HEIF spec says the EXIF orientation tag is to be
    ignored, so it is not applied again here.
    """
    import pillow_heif

    heif = pillow_heif.open_heif(path, convert_hdr_to_8bit=False)
    bits = int(heif.info.get("bit_depth", 8))
    arr = np.asarray(heif)
    if arr.dtype == np.uint16:
        arr = expand_to_16(arr, bits)
    alpha = None
    if arr.shape[2] == 4:
        alpha = np.ascontiguousarray(arr[..., 3])
        arr = arr[..., :3]
    icc = heif.info.get("icc_profile")
    warnings: list[str] = []
    if not icc and heif.info.get("nclx_profile"):
        # Colour described by an nclx box (primaries / transfer, e.g. an HDR
        # PQ or HLG photo) rather than an ICC profile. The pixels pass through
        # unchanged, but the output carries no colour description.
        warnings.append("source colour is described by an nclx box, not an ICC profile; the output has no profile")
    return LoadedImage(
        rgb=np.ascontiguousarray(arr),
        alpha=alpha,
        icc_profile=icc,
        source_mode=f"HEIF {heif.mode}",
        source_bits=bits,
        warnings=warnings,
    )


def _bits_per_channel(path: Path, im: Image.Image) -> int:
    """Bits per channel in a PNG or TIFF file, read from its header.

    Pillow's mode does not tell: a 48-bit RGB PNG opens as plain "RGB".
    """
    if im.format == "PNG":
        with open(path, "rb") as f:
            header = f.read(25)
        return header[24]  # IHDR bit depth: signature 8 + length 4 + type 4 + w 4 + h 4
    if im.format == "TIFF":
        bps = im.tag_v2.get(258, (8,))
        return max(bps) if isinstance(bps, tuple) else int(bps)
    return 8


def _load_deep_with_cv2(path: Path, icc: bytes | None, orientation: int, source_mode: str, bits: int) -> LoadedImage:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # no EXIF rotation applied
    if arr is None:
        raise ValueError(f"OpenCV could not read {path.name}")
    if arr.dtype != np.uint16:
        raise ValueError(f"{path.name}: {arr.dtype} samples are not supported (8- or 16-bit integer only)")
    alpha = None
    if arr.ndim == 2:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 3:
        rgb = arr[..., ::-1]  # OpenCV is BGR
    elif arr.shape[2] == 4:
        rgb = arr[..., 2::-1]
        alpha = apply_orientation(arr[..., 3], orientation)
    else:
        raise ValueError(f"{path.name}: {arr.shape[2]} channels are not supported")
    return LoadedImage(
        rgb=apply_orientation(rgb, orientation),
        alpha=alpha,
        icc_profile=icc,
        source_mode=f"{source_mode} {bits}-bit",
        source_bits=bits,
        warnings=[],
    )


def load_image(path: Path) -> LoadedImage:
    """Read an image as RGB plus optional alpha, 8-bit or 16-bit as the file is.

    The EXIF orientation is applied to the pixels (and so dropped from the
    output). Apps that honour the orientation tag, like PhotoCraft, show the
    original upright; baking the rotation in makes the output line up with
    what they show instead of coming out sideways.

    The ICC profile is kept so colours do not shift when the output is
    placed back over the original. A CMYK source's profile is dropped
    because the pixels have been converted to RGB.
    """
    if path.suffix.lower() in HEIF_SUFFIXES:
        return _load_heif(path)
    with Image.open(path) as im:
        source_mode = im.mode
        icc = im.info.get("icc_profile")
        orientation = im.getexif().get(274, 1)
        bits = _bits_per_channel(path, im)
        if bits > 8 and im.format in ("PNG", "TIFF"):
            return _load_deep_with_cv2(path, icc, orientation, f"{im.format} {source_mode}", bits)
        im.load()
        im = ImageOps.exif_transpose(im)
        if source_mode == "CMYK":
            icc = None
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        alpha = np.asarray(im.convert("RGBA").getchannel("A")).copy() if has_alpha else None
        rgb = np.asarray(im.convert("RGB")).copy()
    return LoadedImage(rgb=rgb, alpha=alpha, icc_profile=icc, source_mode=source_mode, source_bits=8, warnings=[])


def _resize_plane_float(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """LANCZOS-resize one channel through a 32-bit float PIL image.

    Pillow cannot hold 16-bit RGB, but it resizes single-channel float ("F")
    images with the same LANCZOS kernel the 8-bit path uses, so 16-bit images
    are resized channel by channel at full precision.
    """
    return np.asarray(Image.fromarray(plane.astype(np.float32), "F").resize(size, Image.Resampling.LANCZOS))


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """LANCZOS resize to (width, height), keeping the dtype.

    PIL's LANCZOS widens its kernel when shrinking, so a 4x reduction is
    properly anti-aliased rather than just sampled.
    """
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb
    if rgb.dtype == np.uint8:
        return np.asarray(Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.LANCZOS))
    planes = [_resize_plane_float(rgb[..., c], size) for c in range(3)]
    return np.clip(np.rint(np.stack(planes, axis=2)), 0, 65535).astype(np.uint16)


def resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (alpha.shape[1], alpha.shape[0]) == size:
        return alpha
    if alpha.dtype == np.uint8:
        return np.asarray(Image.fromarray(alpha, "L").resize(size, Image.Resampling.LANCZOS))
    return np.clip(np.rint(_resize_plane_float(alpha, size)), 0, 65535).astype(np.uint16)


def _png_insert_iccp(png: bytes, icc: bytes) -> bytes:
    """Add an iCCP chunk (the ICC profile) right after IHDR in an encoded PNG.

    OpenCV, which writes the 16-bit PNGs, has no way to embed a profile.
    iCCP must come before PLTE and IDAT; straight after IHDR satisfies that.
    Any colour chunks the encoder wrote (sRGB, gAMA, cHRM) are dropped, since
    the PNG spec gives the ICC profile the final word and they could only
    contradict it.
    """
    import struct
    import zlib

    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG stream")
    out = [png[:8]]
    pos = 8
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        ctype = png[pos + 4 : pos + 8]
        chunk = png[pos : pos + 12 + length]
        pos += 12 + length
        if ctype in (b"iCCP", b"sRGB", b"gAMA", b"cHRM"):
            continue
        out.append(chunk)
        if ctype == b"IHDR":
            # Profile name, NUL, compression method 0 (zlib), compressed profile.
            data = b"ICC Profile\x00\x00" + zlib.compress(icc)
            crc = zlib.crc32(b"iCCP" + data) & 0xFFFFFFFF
            out.append(struct.pack(">I", len(data)) + b"iCCP" + data + struct.pack(">I", crc))
    return b"".join(out)


def _write_png16(tmp: Path, rgb: np.ndarray, alpha: np.ndarray | None, icc: bytes | None) -> None:
    import cv2

    bgr = rgb[..., ::-1] if alpha is None else np.dstack([rgb[..., ::-1], alpha])
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(bgr), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("16-bit PNG encoding failed")
    data = buf.tobytes()
    tmp.write_bytes(_png_insert_iccp(data, icc) if icc else data)


def _write_tiff16(tmp: Path, rgb: np.ndarray, alpha: np.ndarray | None, icc: bytes | None) -> None:
    import tifffile

    data = rgb if alpha is None else np.dstack([rgb, alpha])
    tifffile.imwrite(
        tmp,
        data,
        photometric="rgb",
        compression="zlib",
        extrasamples=None if alpha is None else ("unassalpha",),
        iccprofile=icc,
    )


def save_image(
    path: Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    icc_profile: bytes | None,
    *,
    jpeg_quality: int = 95,
) -> list[str]:
    """Write the result; the format comes from the file suffix, the depth from the data.

    uint16 data is written as 16-bit PNG or TIFF; JPEG and WebP cannot hold
    it, so for those it is rounded to 8 bits with a warning.

    The image is written to a hidden temporary file in the same folder and
    then renamed over the final name. A rename within one folder is atomic,
    so the final file either does not exist or is complete: anything watching
    for it (a person, or Claude polling the folder) never sees a half-written
    image, and a job killed mid-save leaves only the hidden temp file.

    Returns warnings (for example alpha dropped for JPEG) for the caller.
    """
    warnings: list[str] = []
    suffix = path.suffix.lower()
    if suffix not in OUTPUT_SUFFIXES:
        raise ValueError(f"unsupported output type {suffix!r}; use one of {', '.join(OUTPUT_SUFFIXES)}")
    deep = rgb.dtype == np.uint16
    if deep and suffix in (".jpg", ".jpeg", ".webp"):
        warnings.append(f"{suffix} holds only 8 bits per channel; the 16-bit result was rounded to 8 bits")
        rgb = to_8bit(rgb)
        alpha = to_8bit(alpha) if alpha is not None else None
        deep = False
    if alpha is not None and suffix in (".jpg", ".jpeg"):
        warnings.append("JPEG has no alpha channel; transparency was dropped")
        alpha = None

    tmp = path.with_name(f".{path.name}.partial")
    try:
        if deep and suffix == ".png":
            _write_png16(tmp, rgb, alpha, icc_profile)
        elif deep:  # .tif / .tiff
            _write_tiff16(tmp, rgb, alpha, icc_profile)
        else:
            im = Image.fromarray(rgb, "RGB")
            if alpha is not None:
                im.putalpha(Image.fromarray(alpha, "L"))
            kwargs: dict = {}
            if icc_profile:
                kwargs["icc_profile"] = icc_profile
            if suffix == ".png":
                kwargs["compress_level"] = 6
            elif suffix in (".jpg", ".jpeg"):
                # 4:4:4 chroma: the point of the exercise is fine detail, so do
                # not throw half the colour resolution away on save.
                kwargs.update(quality=jpeg_quality, subsampling=0)
            elif suffix in (".tif", ".tiff"):
                kwargs["compression"] = "tiff_lzw"
            elif suffix == ".webp":
                kwargs["lossless"] = True
            im.save(tmp, format=OUTPUT_FORMATS[suffix], **kwargs)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # only still there if save or rename failed
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
    input_bits: int  # bits per channel in the source file
    output_bits: int  # bits per channel written
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
    warnings = list(image.warnings)
    warnings += save_image(dst, final_rgb, final_alpha, image.icc_profile, jpeg_quality=jpeg_quality)
    if image.alpha is not None:
        warnings.append("alpha channel was resized with LANCZOS, not run through the model")
    written_deep = final_rgb.dtype == np.uint16 and dst.suffix.lower() in (".png", ".tif", ".tiff")
    return UpscaleResult(
        input=str(src),
        output=str(dst),
        input_size=(width, height),
        model_output_size=model_size,
        output_size=final_size,
        input_bits=image.source_bits,
        output_bits=16 if written_deep else 8,
        model=info.as_dict(),
        device=str(device),
        half=half,
        tiles=len(tile_grid(height, width, tile)),
        tile_size=tile,
        tile_overlap=overlap,
        seconds=round(time.perf_counter() - started, 2),
        warnings=warnings,
    )
