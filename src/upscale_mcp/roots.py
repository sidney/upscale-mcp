"""Filesystem boundaries for tool calls.

Same idea as PhotoCraft's automation roots: the server is started with
explicit read roots and one write root, and every path in a tool call must
resolve inside them. Nothing is readable or writable by default.

Paths in tool calls may be absolute (must already be inside a root) or
relative. Relative input paths are taken relative to the *first* read root,
relative output paths relative to the write root. Paths are resolved through
symlinks before the check, so a symlink inside a root that points outside it
is refused rather than followed.
"""

from __future__ import annotations

from pathlib import Path


class RootError(PermissionError):
    pass


class Roots:
    def __init__(self, read_roots: list[Path], write_root: Path | None) -> None:
        # Resolve once at startup so later comparisons are between real paths.
        self.read_roots = [p.expanduser().resolve() for p in read_roots]
        self.write_root = write_root.expanduser().resolve() if write_root else None

    def describe(self) -> dict:
        return {
            "read_roots": [str(p) for p in self.read_roots],
            "write_root": str(self.write_root) if self.write_root else None,
        }

    def input_path(self, value: str) -> Path:
        """Resolve an input image path; it must exist inside a read root."""
        if not self.read_roots:
            raise RootError("no read root configured; start the server with --read-root")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.read_roots[0] / candidate
        try:
            real = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"input not found: {candidate}") from None
        if not real.is_file():
            raise FileNotFoundError(f"input is not a file: {real}")
        if not any(real.is_relative_to(root) for root in self.read_roots):
            raise RootError(f"{real} is outside the read roots {[str(r) for r in self.read_roots]}")
        return real

    def output_path(self, value: str, *, overwrite: bool) -> Path:
        """Resolve an output path inside the write root, creating parent folders.

        Unlike PhotoCraft, missing parent folders are created, but only after
        the target has been checked to lie inside the write root.
        """
        if self.write_root is None:
            raise RootError("no write root configured; start the server with --write-root")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.write_root / candidate
        # Non-strict resolve: the file (and maybe some parents) need not exist
        # yet, but every existing component, symlinks included, is resolved.
        real = candidate.resolve()
        if not real.is_relative_to(self.write_root) or real == self.write_root:
            raise RootError(f"{real} is outside the write root {self.write_root}")
        if real.exists() and not overwrite:
            raise FileExistsError(f"{real} already exists; pass overwrite=true to replace it")
        real.parent.mkdir(parents=True, exist_ok=True)
        return real
