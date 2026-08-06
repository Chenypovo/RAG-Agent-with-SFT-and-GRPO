from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


class ManifestValidationError(ValueError):
    """Raised when prepared evaluation inputs do not match their manifest."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    """Hash relative paths and file contents for a deterministic directory digest."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"directory does not exist: {root}")
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_evaluation_inputs(
    data_dir: str | Path,
    *,
    task_filename: str,
) -> Dict[str, Any]:
    """Validate every prepared input used by an evaluation against manifest.json."""

    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ManifestValidationError(f"data manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot read data manifest: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError(f"data manifest must contain a JSON object: {manifest_path}")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping):
        raise ManifestValidationError("data manifest must contain a 'files' hash mapping")

    required_files = _deduplicate(("tasks.jsonl", task_filename, "corpus.jsonl", "bm25.json"))
    input_hashes: Dict[str, str] = {}
    for filename in required_files:
        expected = declared_files.get(filename)
        if not isinstance(expected, str) or not expected.strip():
            raise ManifestValidationError(f"data manifest is missing a sha256 hash for {filename}")
        input_path = root / filename
        if not input_path.is_file():
            raise ManifestValidationError(f"manifest input does not exist: {input_path}")
        actual = sha256_file(input_path)
        if actual.lower() != expected.strip().lower():
            raise ManifestValidationError(
                f"sha256 mismatch for {filename}: expected {expected}, got {actual}"
            )
        input_hashes[filename] = actual

    return {
        "data_manifest_path": str(manifest_path),
        "data_manifest_sha256": sha256_file(manifest_path),
        "data_input_sha256": input_hashes,
    }


def dependency_versions(distributions: Iterable[str]) -> Dict[str, Any]:
    packages: Dict[str, str | None] = {}
    for distribution in sorted(set(distributions)):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
    }


def ensure_outputs_available(paths: Sequence[str | Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(f"refusing to overwrite existing output(s): {joined}; pass --overwrite")


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
