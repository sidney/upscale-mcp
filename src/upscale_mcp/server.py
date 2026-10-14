"""MCP stdio server exposing open-weight upscaling models.

Written against the MCP Python SDK 2.x, where v1's FastMCP is called
MCPServer (mcp.server.mcpserver). The decorator, Context injection and
report_progress are otherwise the same as v1.

Run by an MCP client (Claude Desktop) as a subprocess that speaks JSON-RPC
over stdin/stdout. Traditionally NOTHING may print to stdout while serving,
or the client sees corrupt protocol messages. SDK 2.x moves the protocol
onto a private file descriptor, which should make a stray print harmless,
but all logging still goes to stderr, which Claude Desktop writes to its MCP
log file.

Tools:
  device_info  - torch version, the device in use, roots, models folders
  list_models  - model files found, with what spandrel reports about each
  upscale      - start one model over one image; wait for it, or not
  job_status   - state of one job, optionally waiting for it to finish
  jobs_list    - recent jobs, newest first
  jobs_cancel  - cancel a queued job, or stop a running one after its tile

Every upscale runs as a background job (see jobs.py) so a caller with a short
call timeout can take the job id and poll, while a caller that can wait gets
the finished result from the same call as before. The handlers stay async
and poll the job, so the server keeps answering while a model runs.

No `from __future__ import annotations` here on purpose: MCPServer builds each
tool's JSON schema, and finds the Context parameter, by inspecting the real
annotation objects. String annotations can defeat that in some SDK versions.
"""

import argparse
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from . import __version__, engine
from .jobs import Job, JobManager
from .roots import Roots

log = logging.getLogger("upscale_mcp")

DEFAULT_MODELS_DIR = Path("~/.local/share/upscale-mcp/models")

# How often a waiting handler checks its job and forwards tile progress.
_POLL_SECONDS = 0.25


class _State:
    """Configuration fixed at startup by main(), shared by the tools."""

    def __init__(self, roots: Roots, models_dirs: list[Path], device, cache: engine.ModelCache) -> None:
        self.roots = roots
        self.models_dirs = models_dirs
        self.device = device
        self.cache = cache
        self.jobs = JobManager()


_state: _State | None = None


def _st() -> _State:
    if _state is None:
        raise RuntimeError("server not configured; start it through main()")
    return _state


# Failures a caller can act on. RootError is a PermissionError; torch reports
# device problems such as "MPS backend out of memory" as RuntimeError.
_ANTICIPATED = (FileNotFoundError, FileExistsError, PermissionError, ValueError, RuntimeError)


@contextmanager
def _as_tool_errors() -> Iterator[None]:
    """Turn anticipated failures into ToolError so their message reaches the client.

    SDK 2.x treats any other exception raised in a tool as a crash: the client
    sees only "Error executing tool <name>" and the traceback goes to the log.
    That is a sensible default for a public server, but here the caller (Claude)
    needs to read "input not found" or "outside the read roots" to fix its next
    call. Exceptions outside _ANTICIPATED still get the crash treatment.
    """
    try:
        yield
    except ToolError:
        raise
    except _ANTICIPATED as exc:
        log.info("tool error: %s: %s", type(exc).__name__, exc)
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc


mcp = MCPServer(
    "upscale-mcp",
    version=__version__,
    instructions=(
        "Runs open-weight image upscaling / restoration models locally on this Mac's GPU. "
        "Call list_models to see which models are installed, then upscale. Input paths must "
        "be inside the configured read roots and outputs go under the write root; relative "
        "paths are taken from the first read root (inputs) and the write root (outputs). "
        "Use output_scale=1 to upscale and then resize back to the original size, which "
        "keeps the model's added detail and lets the result line up pixel for pixel with "
        "the original for compositing. Inputs may be HEIC/HEIF/AVIF and 16-bit PNG/TIFF as well "
        "as the usual formats; images deeper than 8 bits stay 16-bit through the model and are "
        "written as 16-bit PNG/TIFF (JPEG/WebP output is 8-bit). Every upscale runs as a job: if your client gives "
        "up on calls after a minute or so, pass wait_seconds (e.g. 45) and, while the "
        "returned status is queued or running, call job_status with the job id and "
        "wait_seconds again until it is done. AI upscalers invent plausible detail; they "
        "do not recover detail the camera did not capture."
    ),
)


async def _await_job(ctx: Context, job: Job, wait_seconds: float | None) -> dict:
    """Wait for `job` up to `wait_seconds` (None = until finished); return its snapshot.

    While waiting, tile progress is forwarded as MCP progress notifications.
    A job that failed or was cancelled while we waited becomes a ToolError, so
    the call is marked as an error; one still running is returned normally
    with status "queued"/"running" for the caller to poll.
    """
    deadline = None if wait_seconds is None else anyio.current_time() + wait_seconds
    reported = (0, 0)

    async def report() -> None:
        nonlocal reported
        if job.total and (job.done, job.total) != reported:
            reported = (job.done, job.total)
            try:
                await ctx.report_progress(job.done, job.total)
            except Exception:
                pass  # a client that did not ask for progress ignores it

    while not job.finished.is_set():
        if deadline is not None and anyio.current_time() >= deadline:
            break
        await report()
        await anyio.sleep(_POLL_SECONDS)
    # The last tile can finish between two polls; send its update too, so a
    # client watching progress sees the job reach N of N.
    await report()
    snap = job.snapshot()
    if job.finished.is_set() and job.status != "done":
        detail = f": {job.error}" if job.error else ""
        raise ToolError(f"job {job.id} {job.status}{detail}")
    return snap


_WAIT_DESCRIPTION = (
    "Seconds to wait for the job to finish before returning. Omit to wait until it is done. "
    "0 returns the job id at once. A client that times out calls after about a minute "
    "should pass about 45 and poll job_status while the status is queued or running."
)


@mcp.tool()
def device_info() -> dict:
    """Report the torch version, the compute device in use, the roots and the models folders."""
    import torch

    st = _st()
    return {
        "torch": torch.__version__,
        "device": str(st.device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "models_dirs": [str(p) for p in st.models_dirs],
        **st.roots.describe(),
    }


@mcp.tool()
async def list_models(
    inspect: Annotated[
        bool,
        Field(
            description=(
                "Load each file on the CPU to report its architecture, scale and channels. "
                "The first listing of a large collection can take a few seconds; results are "
                "cached until a file changes."
            )
        ),
    ] = True,
) -> dict:
    """List the model files in the models folders."""
    st = _st()

    def work() -> dict:
        models: list[dict] = []
        for path in engine.find_model_files(st.models_dirs):
            entry: dict = {"name": path.name, "path": str(path)}
            if inspect:
                try:
                    entry.update(st.cache.info(path).as_dict())
                except Exception as exc:  # unsupported architecture, corrupt file, ...
                    entry["error"] = f"{type(exc).__name__}: {exc}"
            models.append(entry)
        return {"models_dirs": [str(p) for p in st.models_dirs], "models": models}

    return await anyio.to_thread.run_sync(work)


def _default_output_name(src: Path, model_path: Path, output_scale: float | None) -> str:
    """e.g. IMG_1234.RealESRGAN_x4plus.x1.png: says which model and final scale."""
    scale = "native" if output_scale is None else f"x{output_scale:g}"
    return f"{src.stem}.{model_path.stem}.{scale}.png"


@mcp.tool()
async def upscale(
    ctx: Context,
    input: Annotated[
        str,
        Field(
            description=(
                "Input image: absolute path inside a read root, or relative to the first read root. "
                "Anything Pillow reads, plus HEIC/HEIF/AVIF and 16-bit PNG/TIFF (kept at 16 bits)."
            )
        ),
    ],
    model: Annotated[
        str,
        Field(description="Model file name or stem from list_models, e.g. 'RealESRGAN_x4plus'."),
    ],
    output: Annotated[
        str | None,
        Field(
            description=(
                "Output path (absolute inside the write root, or relative to it). The suffix picks "
                "the format: .png .jpg .tif .webp; .png and .tif keep 16 bits for a deeper source. "
                "Default: <input stem>.<model stem>.<scale>.png in the write root."
            )
        ),
    ] = None,
    output_scale: Annotated[
        float | None,
        Field(
            gt=0,
            le=16,
            description=(
                "Final size relative to the INPUT. Omit to keep the model's native scale (4x for a "
                "4x model). 1 = resize back to the input size (for compositing over the original). "
                "Other values resize to that multiple with LANCZOS."
            ),
        ),
    ] = None,
    tile_size: Annotated[
        int,
        Field(ge=0, le=4096, description="Tile edge in input pixels; 0 = whole image in one pass."),
    ] = 512,
    tile_overlap: Annotated[
        int,
        Field(ge=0, le=256, description="Context margin around each tile, in input pixels, to hide seams."),
    ] = 32,
    half: Annotated[
        bool,
        Field(description="Run in float16 on the GPU (faster, less memory) if the model supports it."),
    ] = False,
    jpeg_quality: Annotated[int, Field(ge=1, le=100, description="Quality for .jpg output.")] = 95,
    overwrite: Annotated[bool, Field(description="Replace an existing output file.")] = False,
    wait_seconds: Annotated[float | None, Field(ge=0, le=3600, description=_WAIT_DESCRIPTION)] = None,
) -> dict:
    """Upscale (or restore) one image with one model, writing the result under the write root.

    Returns the job's status. When the job is done, `result` holds the output
    path, sizes, model details, device and timing.
    """
    with _as_tool_errors():
        st = _st()
        # Validate everything that can be checked up front, so mistakes come
        # back immediately instead of as a failed job.
        src = st.roots.input_path(input)
        model_path = engine.resolve_model(model, st.models_dirs)
        dst = st.roots.output_path(
            output if output else _default_output_name(src, model_path, output_scale),
            overwrite=overwrite,
        )

        def work(progress) -> dict:
            log.info("upscale %s with %s -> %s on %s", src, model_path.name, dst, st.device)
            result = engine.upscale_file(
                src,
                dst,
                model_path,
                st.cache,
                st.device,
                output_scale=output_scale,
                tile=tile_size,
                overlap=tile_overlap,
                half=half,
                jpeg_quality=jpeg_quality,
                progress=progress,
            )
            log.info("done in %.1fs", result.seconds)
            return result.as_dict()

        job = st.jobs.submit(input=str(src), model=model_path.name, output=dst, work=work)
        return await _await_job(ctx, job, wait_seconds)


@mcp.tool()
async def job_status(
    ctx: Context,
    job: Annotated[str, Field(description="Job id returned by upscale, e.g. 'j3'.")],
    wait_seconds: Annotated[
        float,
        Field(ge=0, le=3600, description="Seconds to wait for the job to finish before answering; 0 = answer now."),
    ] = 0,
) -> dict:
    """Report one job's status, progress and, once done, its result."""
    with _as_tool_errors():
        return await _await_job(ctx, _st().jobs.get(job), wait_seconds)


@mcp.tool()
def jobs_list() -> dict:
    """List recent jobs (queued, running and the last finished ones), newest first."""
    return {"jobs": _st().jobs.list()}


@mcp.tool()
def jobs_cancel(
    job: Annotated[str, Field(description="Job id to cancel.")],
) -> dict:
    """Cancel a queued job, or stop a running one after the tile it is working on.

    A running job stops at the next tile boundary, so the returned status may
    still be "running"; check job_status. A cancelled job writes no output.
    """
    with _as_tool_errors():
        return _st().jobs.cancel(job).snapshot()


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------


def _self_check(models_dirs: list[Path], device, cache: engine.ModelCache, model_name: str | None) -> int:
    """`--check`: a first-run test to run in Terminal before wiring up Claude Desktop.

    Prints the torch/device situation and the models found. If there is a
    model, runs it on a small synthetic image on the chosen device AND on the
    CPU and reports the largest per-pixel difference between the two (in
    0..255 levels). A large difference means the GPU backend is computing
    something different from the CPU reference, for example a broken MPS
    kernel; a few levels is ordinary floating-point noise.
    """
    import time

    import numpy as np
    import torch

    print(f"torch {torch.__version__}; device {device}; mps available: {torch.backends.mps.is_available()}")
    files = engine.find_model_files(models_dirs)
    print(f"models dirs: {', '.join(str(p) for p in models_dirs)}")
    if not files:
        print("no model files found; put a .pth or .safetensors upscaler in a models dir")
        return 1
    for path in files:
        try:
            info = cache.info(path)
            print(f"  {path.name}: {info.architecture}, {info.purpose}, x{info.scale}, half={info.supports_half}")
        except Exception as exc:
            print(f"  {path.name}: cannot load ({type(exc).__name__}: {exc})")

    model_path = engine.resolve_model(model_name, models_dirs) if model_name else files[0]
    rng = np.random.default_rng(0)
    # Smooth-ish synthetic content (blurred noise) so the model has edges and
    # gradients to work on, not just white noise.
    base = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    from PIL import Image

    img = np.asarray(Image.fromarray(base).resize((96, 96), Image.Resampling.BICUBIC))

    results = {}
    for dev in dict.fromkeys([device, torch.device("cpu")]):  # de-duplicated, order kept
        desc, info = cache.get(model_path, dev, half=False)
        t0 = time.perf_counter()
        results[str(dev)] = engine.upscale_array(img, desc, info.scale, desc.size_requirements, dev, tile=64, overlap=16)
        print(f"ran {model_path.name} on {dev}: {time.perf_counter() - t0:.2f}s for a 96x96 test image")
    if len(results) == 2:
        a, b = results.values()
        diff = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
        print(f"max difference {device} vs cpu: {diff} levels (0..255)")
        if diff > 8:
            print("WARNING: the GPU result differs noticeably from the CPU reference")
            return 2
    print("ok")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="upscale-mcp",
        description="MCP stdio server that runs open-weight upscaling models (via spandrel) locally.",
    )
    parser.add_argument(
        "--models-dir",
        action="append",
        type=Path,
        help=f"folder of model files, searched recursively; repeatable (default {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--read-root", action="append", type=Path, default=[], help="folder inputs may come from; repeatable")
    parser.add_argument("--write-root", type=Path, help="folder outputs are written under")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--check", action="store_true", help="run a self-test in the terminal and exit; does not serve")
    parser.add_argument("--check-model", help="model to use for --check (default: the first one found)")
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    models_dirs = [p.expanduser().resolve() for p in (args.models_dir or [DEFAULT_MODELS_DIR])]
    device = engine.pick_device(args.device)
    cache = engine.ModelCache()

    if args.check:
        sys.exit(_self_check(models_dirs, device, cache, args.check_model))

    roots = Roots(args.read_root, args.write_root)
    for d in models_dirs:
        if not d.is_dir():
            log.warning("models dir %s does not exist", d)
    if roots.write_root and not roots.write_root.is_dir():
        log.warning("write root %s does not exist", roots.write_root)

    global _state
    _state = _State(roots, models_dirs, device, cache)
    log.info("serving on stdio; device %s; models %s; roots %s", device, models_dirs, roots.describe())
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
