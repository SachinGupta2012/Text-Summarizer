"""Create a compact Phase 3 artifact for transfer into Phase 4."""

import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.policy import Phase3Manifest, PolicyDecision


class Phase3HandoffManifest(BaseModel):
    """Contents and provenance of the compact Phase 3 archive."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    phase3_manifest_sha256: str
    training_ready_records: int
    routed_source_records: int
    included_files: int


class Phase3HandoffResult(BaseModel):
    """Verified archive identity returned by handoff creation."""

    model_config = ConfigDict(frozen=True)

    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    manifest: Phase3HandoffManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_route_ids(path: Path) -> set[str]:
    record_ids: set[str] = set()
    with path.open(encoding="utf-8") as route_file:
        for line_number, line in enumerate(route_file, start=1):
            if not line.strip():
                continue
            try:
                decision = PolicyDecision.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid chunking route at {path}:{line_number}") from error
            if decision.record_id in record_ids:
                raise ValueError("Chunking route contains duplicate record IDs.")
            record_ids.add(decision.record_id)
    return record_ids


def _materialize_routed_sources(
    canonical_root: Path,
    destination_root: Path,
    routed_record_ids: set[str],
) -> int:
    found: set[str] = set()
    for source_path in sorted(canonical_root.glob("*/*.jsonl")):
        relative_path = source_path.relative_to(canonical_root)
        destination_path = destination_root / relative_path
        output = None
        try:
            with source_path.open(encoding="utf-8") as source_file:
                for line in source_file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_id = str(record["record_id"])
                    if record_id not in routed_record_ids:
                        continue
                    if output is None:
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        output = destination_path.open("x", encoding="utf-8", newline="\n")
                    output.write(line if line.endswith("\n") else f"{line}\n")
                    found.add(record_id)
        finally:
            if output is not None:
                output.close()
    missing = routed_record_ids.difference(found)
    if missing:
        raise RuntimeError(f"Canonical data is missing {len(missing)} routed records.")
    return len(found)


def _add_file(
    archive: tarfile.TarFile,
    path: Path,
    archive_name: Path,
) -> None:
    archive.add(path, arcname=archive_name.as_posix(), recursive=False)


def create_phase3_handoff(
    final_directory: Path,
    archive_path: Path,
) -> Phase3HandoffResult:
    """Archive only Phase 4 inputs without modifying Phase 3 artifacts."""

    if archive_path.exists():
        raise FileExistsError(f"Phase 3 handoff archive already exists: {archive_path}")
    training_ready = final_directory / "training-ready"
    phase3_manifest_path = training_ready / "phase3-manifest.json"
    route_root = training_ready / "routing"
    required_paths = (
        phase3_manifest_path,
        route_root / "requires_chunking.jsonl",
        route_root / "long_target.jsonl",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing or not (training_ready / "processed").is_dir():
        raise FileNotFoundError(f"Phase 3 handoff inputs are missing: {missing}")

    phase3_manifest = Phase3Manifest.model_validate_json(
        phase3_manifest_path.read_text(encoding="utf-8")
    )
    route_ids = _load_route_ids(route_root / "requires_chunking.jsonl")
    if len(route_ids) != phase3_manifest.by_action.get("requires_chunking", 0):
        raise RuntimeError("Chunking route count does not match the Phase 3 manifest.")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase3-handoff-",
        dir=archive_path.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        routed_root = temporary_root / "processed"
        routed_count = _materialize_routed_sources(
            final_directory / "processed",
            routed_root,
            route_ids,
        )
        training_paths = tuple(
            sorted(
                path
                for path in training_ready.rglob("*")
                if path.is_file()
                and (
                    path == phase3_manifest_path
                    or "processed" in path.relative_to(training_ready).parts
                    or path.parent == route_root
                )
            )
        )
        routed_paths = tuple(sorted(path for path in routed_root.rglob("*") if path.is_file()))
        included_files = len(training_paths) + len(routed_paths) + 1
        handoff_manifest = Phase3HandoffManifest(
            phase3_manifest_sha256=_sha256(phase3_manifest_path),
            training_ready_records=phase3_manifest.training_ready_records,
            routed_source_records=routed_count,
            included_files=included_files,
        )
        handoff_manifest_path = temporary_root / "handoff-manifest.json"
        handoff_manifest_path.write_text(
            json.dumps(
                handoff_manifest.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        with tarfile.open(archive_path, mode="w:gz", compresslevel=3) as archive:
            for path in training_paths:
                _add_file(
                    archive,
                    path,
                    Path("phase3-handoff") / "training-ready" / path.relative_to(training_ready),
                )
            for path in routed_paths:
                _add_file(
                    archive,
                    path,
                    Path("phase3-handoff") / "processed" / path.relative_to(routed_root),
                )
            _add_file(
                archive,
                handoff_manifest_path,
                Path("phase3-handoff") / "handoff-manifest.json",
            )

    with tarfile.open(archive_path, mode="r:gz") as archive:
        file_members = tuple(member for member in archive.getmembers() if member.isfile())
    if len(file_members) != handoff_manifest.included_files:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("Phase 3 handoff archive verification failed.")
    return Phase3HandoffResult(
        archive_path=archive_path,
        archive_bytes=archive_path.stat().st_size,
        archive_sha256=_sha256(archive_path),
        manifest=handoff_manifest,
    )
