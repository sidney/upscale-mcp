"""Tests that need no model weights and no GPU.

The tiling test uses a tiny stand-in model whose receptive field is known
exactly (nearest-neighbour upscale followed by a 3x3 convolution, so each
output pixel depends on at most one input pixel in every direction). With an
overlap of at least that, tiled and single-pass outputs must be identical,
bit for bit. Real models have far larger receptive fields, so for them the
overlap only makes the seams invisible, not mathematically absent; the
README's numbers for Real-ESRGAN come from a separate measurement.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from upscale_mcp import engine
from upscale_mcp.roots import RootError, Roots


@dataclass
class Req:
    minimum: int = 0
    multiple_of: int = 1
    square: bool = False


class TinyModel(torch.nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.up = torch.nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(self.up(x)))


def _image(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(1).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_tiled_matches_single_pass(scale: int) -> None:
    img = _image(70, 53)  # deliberately not a multiple of the tile size
    model = TinyModel(scale).eval()
    cpu = torch.device("cpu")
    whole = engine.upscale_array(img, model, scale, Req(), cpu, tile=0)
    tiled = engine.upscale_array(img, model, scale, Req(), cpu, tile=16, overlap=2)
    assert whole.shape == (70 * scale, 53 * scale, 3)
    assert np.array_equal(whole, tiled)


def test_padding_for_size_requirements_is_cropped_off() -> None:
    img = _image(37, 29)
    model = TinyModel(2).eval()
    req = Req(minimum=32, multiple_of=8, square=True)
    out = engine.upscale_array(img, model, 2, req, torch.device("cpu"), tile=16, overlap=4)
    assert out.shape == (74, 58, 3)


def test_progress_reports_every_tile() -> None:
    calls: list[tuple[int, int]] = []
    engine.upscale_array(
        _image(40, 40), TinyModel(1).eval(), 1, Req(), torch.device("cpu"), tile=16, overlap=2,
        progress=lambda d, t: calls.append((d, t)),
    )
    assert calls == [(i, 9) for i in range(1, 10)]


def test_roots_refuse_escapes(tmp_path: Path) -> None:
    read = tmp_path / "pictures"
    write = read / "out"
    outside = tmp_path / "elsewhere"
    for d in (read, write, outside):
        d.mkdir()
    (read / "a.png").write_bytes(b"x")
    (outside / "secret.png").write_bytes(b"x")
    (read / "link.png").symlink_to(outside / "secret.png")
    roots = Roots([read], write)

    assert roots.input_path("a.png") == (read / "a.png").resolve()
    assert roots.input_path(str(read / "a.png")) == (read / "a.png").resolve()
    with pytest.raises(RootError):
        roots.input_path("../elsewhere/secret.png")
    with pytest.raises(RootError):
        roots.input_path(str(outside / "secret.png"))
    with pytest.raises(RootError):
        roots.input_path("link.png")  # symlink pointing outside

    assert roots.output_path("sub/b.png", overwrite=False) == (write / "sub" / "b.png").resolve()
    assert (write / "sub").is_dir()
    with pytest.raises(RootError):
        roots.output_path("../a2.png", overwrite=False)  # inside read root, outside write root
    (write / "c.png").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        roots.output_path("c.png", overwrite=False)
    assert roots.output_path("c.png", overwrite=True).name == "c.png"


def test_roots_fail_closed() -> None:
    roots = Roots([], None)
    with pytest.raises(RootError):
        roots.input_path("a.png")
    with pytest.raises(RootError):
        roots.output_path("a.png", overwrite=False)


def test_resolve_model_names(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "4x_A.pth").write_bytes(b"")
    (tmp_path / "sub" / "B.safetensors").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    dirs = [tmp_path]
    assert engine.resolve_model("4x_a", dirs).name == "4x_A.pth"
    assert engine.resolve_model("B.safetensors", dirs).name == "B.safetensors"
    assert engine.resolve_model("sub/b.safetensors", dirs).name == "B.safetensors"
    with pytest.raises(FileNotFoundError):
        engine.resolve_model("notes", dirs)
    with pytest.raises(ValueError):
        engine.resolve_model("../4x_A.pth", dirs)
