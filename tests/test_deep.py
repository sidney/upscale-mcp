"""16-bit and HEIF handling: no model weights or GPU needed."""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms

from upscale_mcp import engine

# A real ICC profile to carry through: Pillow's built-in sRGB.
ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@dataclass
class Req:
    minimum: int = 0
    multiple_of: int = 1
    square: bool = False


class TinyModel(torch.nn.Module):
    """Nearest-neighbour upscale plus a 3x3 conv: a known receptive field."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.up = torch.nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(self.up(x)))


def _deep(h: int, w: int, channels: int = 3) -> np.ndarray:
    shape = (h, w, channels) if channels > 1 else (h, w)
    return np.random.default_rng(2).integers(0, 65536, size=shape, dtype=np.uint16)


def test_expand_to_16_maps_white_to_white() -> None:
    codes = np.array([0, 1, 512, 1023], dtype=np.uint16) << 6  # 10-bit, left-aligned like pillow-heif
    out = engine.expand_to_16(codes, 10)
    assert out.tolist() == [0, 64, round(512 * 65535 / 1023), 65535]
    assert engine.expand_to_16(codes, 16) is codes


def test_to_8bit_rounds() -> None:
    assert engine.to_8bit(np.array([0, 128, 129, 257 * 200, 65535], dtype=np.uint16)).tolist() == [0, 0, 1, 200, 255]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_matches_pillow(orientation: int) -> None:
    a = np.random.default_rng(3).integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    # `is not None`: FLIP_LEFT_RIGHT is enum value 0, which is falsy.
    expected = np.asarray(Image.fromarray(a).transpose(method)) if method is not None else a
    assert np.array_equal(engine.apply_orientation(a, orientation), expected)


def test_tiled_16bit_matches_single_pass() -> None:
    img = _deep(40, 33)
    model = TinyModel(2).eval()
    cpu = torch.device("cpu")
    whole = engine.upscale_array(img, model, 2, Req(), cpu, tile=0)
    tiled = engine.upscale_array(img, model, 2, Req(), cpu, tile=16, overlap=2)
    assert whole.dtype == np.uint16 and whole.shape == (80, 66, 3)
    assert np.array_equal(whole, tiled)


def test_8bit_and_16bit_paths_agree() -> None:
    """The same picture at both depths should give the same result, to 8-bit precision."""
    img8 = np.random.default_rng(4).integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    img16 = img8.astype(np.uint16) * 257
    model = TinyModel(2).eval()
    cpu = torch.device("cpu")
    out8 = engine.upscale_array(img8, model, 2, Req(), cpu, tile=0)
    out16 = engine.upscale_array(img16, model, 2, Req(), cpu, tile=0)
    assert np.abs(engine.to_8bit(out16).astype(int) - out8.astype(int)).max() <= 1


@pytest.mark.parametrize("suffix", [".png", ".tif"])
@pytest.mark.parametrize("with_alpha", [False, True])
def test_16bit_round_trip_keeps_values_and_profile(tmp_path: Path, suffix: str, with_alpha: bool) -> None:
    rgb = _deep(9, 13)
    alpha = _deep(9, 13, channels=1) if with_alpha else None
    path = tmp_path / f"out{suffix}"
    assert engine.save_image(path, rgb, alpha, ICC) == []
    loaded = engine.load_image(path)
    assert loaded.source_bits == 16
    assert loaded.rgb.dtype == np.uint16 and np.array_equal(loaded.rgb, rgb)
    if with_alpha:
        assert np.array_equal(loaded.alpha, alpha)
    else:
        assert loaded.alpha is None
    assert loaded.icc_profile == ICC
    # An independent reader sees the profile too.
    with Image.open(path) as im:
        assert im.info.get("icc_profile") == ICC
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]  # no temp left


def test_16bit_png_header_says_16_bits(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    engine.save_image(path, _deep(4, 4), None, ICC)
    assert path.read_bytes()[24] == 16


def test_16bit_to_jpeg_rounds_with_warning(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    warnings = engine.save_image(path, _deep(8, 8), None, ICC)
    assert any("8 bits" in w for w in warnings)
    assert engine.load_image(path).rgb.dtype == np.uint8


def test_8bit_png_still_loads_as_8bit(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    a = np.random.default_rng(5).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    Image.fromarray(a).save(path, icc_profile=ICC)
    loaded = engine.load_image(path)
    assert loaded.source_bits == 8 and loaded.rgb.dtype == np.uint8 and np.array_equal(loaded.rgb, a)


def test_heif_10bit_loads_as_full_range_16bit(tmp_path: Path) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    # A smooth 10-bit ramp, written as a 10-bit HEIF with an ICC profile.
    h, w = 64, 64
    ramp = np.linspace(0, 1023, w, dtype=np.float64)
    codes = np.repeat(np.repeat(ramp[None, :, None], h, axis=0), 3, axis=2).round().astype(np.uint16)
    heif = pillow_heif.from_bytes(mode="RGB;16", size=(w, h), data=(codes << 6).tobytes())
    heif.info["icc_profile"] = ICC
    path = tmp_path / "ramp.heic"
    try:
        heif.save(path, quality=-1)  # lossless where the encoder allows it
    except Exception as exc:  # no HEVC encoder in this build of libheif
        pytest.skip(f"cannot write HEIF here: {exc}")
    loaded = engine.load_image(path)
    assert loaded.source_bits == 10
    assert loaded.rgb.dtype == np.uint16
    assert loaded.icc_profile == ICC
    expected = engine.expand_to_16(codes << 6, 10).astype(int)
    # Lossless RGB HEVC should round-trip; allow a few 10-bit codes for
    # encoders that go through YCbCr anyway.
    assert np.abs(loaded.rgb.astype(int) - expected).max() <= 4 * 64
