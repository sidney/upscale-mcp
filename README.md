# upscale-mcp

This is an MCP server I (well, Claude, actually) wrote for my personal use
on my MacBook Pro M5 Pro, but there's no reason not to make it public.

I release it under the Apache 2.0 license.

All copyrightable files in this project copyright 2026 Sidney Markowitz

The MCP server runs open-weight image upscaling and restoration models
locally, on the Mac's GPU through PyTorch's Metal (MPS) backend. Models are
loaded with [spandrel](https://github.com/chaiNNer-org/spandrel), the loader
chaiNNer uses, so the usual community model files work.

It is meant to be driven by an MCP client such as Claude Desktop: list the
installed models, run one over an image, get the result written to a folder.
With `output_scale=1` the image is upscaled and then resized back to its
original size, so the result lines up pixel for pixel with the original and
can be composited over it (for example masked to one subject in an editor).

AI upscalers invent plausible detail. They do not recover detail the camera
did not capture, and different models invent very differently, so it pays to
compare a few on the image at hand.

## Install

Requires [uv](https://docs.astral.sh/uv/). `.python-version` pins Python 3.12,
which uv installs if needed.

```sh
cd ~/OpenSource/upscale-mcp
uv sync
```

## Models

Model files live outside the repo. By default the server looks in
`~/.local/share/upscale-mcp/models`; pass `--models-dir` (repeatable) to use
other folders, for example ComfyUI's `models/upscale_models`. Folders are
searched recursively. Supported file types: `.pth`, `.pt`, `.safetensors`,
`.ckpt`. Pickled `.pth`/`.pt`/`.ckpt` files are loaded by spandrel with torch's
restricted `weights_only` unpickler.

A reasonable baseline is Real-ESRGAN x4plus (BSD-3-Clause):

    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    sha256 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1

It is trained to remove heavy degradation and tends to smooth fine texture
such as fur into a painted look. [OpenModelDB](https://openmodeldb.info/)
catalogues many alternatives, filterable by purpose (photo, JPEG, blur, ...).
Some architectures that spandrel supports only through the separate
`spandrel_extra_arches` package (non-commercial licences) are not included.

## First run

Before adding the server to Claude Desktop, run the self-test in Terminal:

```sh
uv run upscale-mcp --check
```

It prints the torch version and device, lists the models it finds, then runs
the first model (or `--check-model NAME`) on a small synthetic image on the
GPU and on the CPU and reports the largest difference between the two. A few
levels (out of 255) is floating-point noise; a large difference means the
Metal backend is computing something different from the CPU.

## Claude Desktop

Add to `mcpServers` in
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
"upscale": {
  "command": "/Users/sidney/.local/bin/uv",
  "args": [
    "--directory", "/Users/sidney/OpenSource/upscale-mcp",
    "run", "upscale-mcp",
    "--models-dir", "/Users/sidney/.local/share/upscale-mcp/models",
    "--read-root", "/Users/sidney/Pictures",
    "--write-root", "/Users/sidney/Pictures/Photocraft"
  ]
}
```

## Tools

- `device_info`: torch version, device in use, roots, models folders.
- `list_models`: model files with architecture, purpose, scale and whether
  they support half precision.
- `upscale`: run one model over one image, as a background job.
  - `input`: inside a read root; relative paths start at the first read root.
  - `model`: file name or stem from `list_models`.
  - `output`: inside the write root (relative paths start there); the suffix
    picks the format (`.png`, `.jpg`, `.tif`, `.webp`). Default
    `<input stem>.<model stem>.<scale>.png`. Existing files are not
    replaced unless `overwrite` is true.
  - `output_scale`: omit for the model's native scale, `1` for the original
    size, or any other multiple of the input size (LANCZOS resize).
  - `tile_size` (default 512, 0 = one pass), `tile_overlap` (default 32),
    `half` (float16 on the GPU), `jpeg_quality`.
  - `wait_seconds`: omit to wait until the job is done and get the result
    from the same call; `0` to get the job id back at once; anything else to
    wait up to that long. The reply is the job's status, with the result
    under `result` once it is `done`.
- `job_status`: one job's status, tile progress and, when done, its result.
  `wait_seconds` (default 0) waits up to that long for it to finish first.
- `jobs_list`: recent jobs, newest first.
- `jobs_cancel`: cancel a queued job, or stop a running one after its current
  tile. A cancelled job writes no output.

Jobs run one at a time on the GPU. Some clients give
up on a call long before a slow model finishes (a Claude cloud session
reaching the Mac through the desktop app waits 60 seconds), so such a client
should pass `wait_seconds` of about 45 and keep calling `job_status` while
the status is `queued` or `running`. A job shows 0 of 0 tiles while its model
is loading.

Outputs are written to a hidden temporary file and renamed into place, so a
file that exists under its final name is always complete.

Inputs can be anything Pillow reads, plus HEIC/HEIF/AVIF (through
pillow-heif) and 16-bit PNG and TIFF. Images deeper than 8 bits, such as a
phone's 10-bit HEIC, are processed at 16 bits throughout and written as
16-bit PNG or TIFF. JPEG and WebP output is always 8-bit, with a warning if
that rounds a deeper result. The reply reports `input_bits` and `output_bits`.

The EXIF orientation is applied to the pixels and the ICC profile is copied
to the output. Alpha is carried through with a plain LANCZOS resize, not the
model.

## Paths

Nothing is readable or writable unless granted at startup with `--read-root`
(repeatable) and `--write-root`. Paths are resolved through symlinks before
the check, so a symlink inside a root that points outside it is refused.
Missing folders under the write root are created.

## Tiling

Large images are processed in tiles so GPU memory depends on the tile size,
not the image size. Each tile is run with `tile_overlap` pixels of extra
context on every side and only its centre is kept. Measured with Real-ESRGAN
x4plus on a 256x256 crop, 128-pixel tiles against a single pass: overlap 16
left 1.2% of pixels more than 2 levels off (max 9); overlap 32 left 0.065%
(max 8), which is not visible.

## Tests

```sh
uv run pytest
```

The tests need no model weights or GPU: they check that tiled and
single-pass output are identical for a model with a known receptive field,
the padding for architectures with size requirements, progress reporting,
path confinement, model-name resolution, the job queue (one at a time,
cancelling queued and running jobs, refusing two jobs on one output),
atomic saves, and the 16-bit path (10-bit HEIF scaling, EXIF orientation,
16-bit PNG and TIFF round trips with their ICC profiles).
