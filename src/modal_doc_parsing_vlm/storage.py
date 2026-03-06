from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from .types_result import (
    ChunkParseSummary,
    IdempotencyRecord,
    JobManifest,
    JobProgressSnapshot,
    PageChunk,
    PageParseResult,
    PageTask,
    QualityStage,
    ResultLevel,
)

T = TypeVar("T", bound=BaseModel)


class KeyValueStore(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool | None: ...


class InMemoryKVStore:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self._data:
            return False
        self._data[key] = value
        return True


class StorageBackend(Protocol):
    def create_job_manifest(self, manifest: JobManifest) -> None: ...

    def write_page_task(self, task: PageTask) -> None: ...

    def write_page_result(self, result: PageParseResult) -> None: ...

    def read_page_result(self, job_id: str, page_id: int) -> PageParseResult | None: ...

    def write_final_result(self, job_id: str, result, markdown: str, text: str) -> None: ...

    def get_status(self, job_id: str) -> JobProgressSnapshot | None: ...

    def set_status(self, job_id: str, snapshot: JobProgressSnapshot) -> None: ...

    def lookup_idempotency(self, request_fingerprint: str) -> IdempotencyRecord | None: ...

    def store_idempotency(self, record: IdempotencyRecord) -> None: ...


class FileSystemStorageBackend:
    def __init__(
        self,
        root_path: Path,
        *,
        status_store: KeyValueStore | None = None,
        idempotency_store: KeyValueStore | None = None,
        volume: Any | None = None,
    ) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.status_store = status_store or InMemoryKVStore()
        self.idempotency_store = idempotency_store or InMemoryKVStore()
        self.volume = volume
        self._batch_depth = 0
        self._batch_dirty = False

    def commit(self) -> None:
        if self.volume is not None:
            self.volume.commit()

    def reload(self) -> None:
        if self.volume is not None:
            self.volume.reload()

    def _commit_if_needed(self) -> None:
        if self._batch_depth > 0:
            self._batch_dirty = True
            return
        self.commit()

    @contextmanager
    def batch(self) -> Iterator[None]:
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_dirty:
                self._batch_dirty = False
                self.commit()

    def _write_json(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        temp_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)
        self._commit_if_needed()

    def _read_model(self, path: Path, model_type: type[T]) -> T | None:
        if not path.exists():
            return None
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_model(self, path: Path, model: BaseModel) -> None:
        self._write_json(path, model.model_dump(mode="json"))

    def job_dir(self, job_id: str) -> Path:
        return self.root_path / "jobs" / job_id

    def upload_dir(self, upload_id: str) -> Path:
        return self.root_path / "uploads" / upload_id

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def source_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "source" / "original"

    def page_task_path(self, job_id: str, page_id: int) -> Path:
        return self.job_dir(job_id) / "pages" / str(page_id) / "task.json"

    def page_result_path(self, job_id: str, page_id: int) -> Path:
        return self.job_dir(job_id) / "pages" / str(page_id) / "result.json"

    def page_result_revision_path(self, job_id: str, page_id: int, result_revision: int) -> Path:
        return self.job_dir(job_id) / "pages" / str(page_id) / f"result.r{result_revision}.json"

    def page_image_path(self, job_id: str, page_id: int) -> Path:
        return self.job_dir(job_id) / "pages" / str(page_id) / "page.png"

    def chunk_path(self, job_id: str, chunk_id: str) -> Path:
        return self.job_dir(job_id) / "chunks" / f"{chunk_id}.json"

    def final_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result" / "document_parse_result.json"

    def final_stage_json_path(self, job_id: str, stage: QualityStage) -> Path:
        return self.job_dir(job_id) / "result" / f"document_parse_result.{stage.value}.json"

    def final_stage_revision_json_path(
        self, job_id: str, stage: QualityStage, result_revision: int
    ) -> Path:
        return (
            self.job_dir(job_id)
            / "result"
            / f"document_parse_result.{stage.value}.r{result_revision}.json"
        )

    def final_markdown_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result" / "document.md"

    def final_stage_markdown_path(self, job_id: str, stage: QualityStage) -> Path:
        return self.job_dir(job_id) / "result" / f"document.{stage.value}.md"

    def final_text_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result" / "document.txt"

    def final_stage_text_path(self, job_id: str, stage: QualityStage) -> Path:
        return self.job_dir(job_id) / "result" / f"document.{stage.value}.txt"

    def create_job_manifest(self, manifest: JobManifest) -> None:
        self.write_job_manifest(manifest)

    def write_job_manifest(self, manifest: JobManifest) -> None:
        self._write_model(self.manifest_path(manifest.job_id), manifest)

    def read_job_manifest(self, job_id: str) -> JobManifest:
        manifest = self._read_model(self.manifest_path(job_id), JobManifest)
        if manifest is None:
            raise FileNotFoundError(f"Unknown job_id: {job_id}")
        return manifest

    def write_source_bytes(self, job_id: str, data: bytes) -> None:
        path = self.source_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self._commit_if_needed()

    def read_source_bytes(self, job_id: str) -> bytes:
        return self.source_path(job_id).read_bytes()

    def write_upload(
        self,
        upload_id: str,
        data: bytes,
        *,
        file_name: str,
        mime_type: str | None = None,
    ) -> None:
        upload_dir = self.upload_dir(upload_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / "blob").write_bytes(data)
        metadata = {"file_name": file_name}
        if mime_type is not None:
            metadata["mime_type"] = mime_type
        (upload_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._commit_if_needed()

    def read_upload(self, upload_id: str) -> dict[str, Any] | None:
        metadata_path = self.upload_dir(upload_id) / "metadata.json"
        blob_path = self.upload_dir(upload_id) / "blob"
        if not metadata_path.exists() or not blob_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"data": blob_path.read_bytes(), **metadata}

    def write_chunk_manifest(self, chunk: PageChunk) -> None:
        self._write_model(self.chunk_path(chunk.job_id, chunk.chunk_id), chunk)

    def read_chunk_manifest(self, job_id: str, chunk_id: str) -> PageChunk | None:
        return self._read_model(self.chunk_path(job_id, chunk_id), PageChunk)

    def write_page_task(self, task: PageTask) -> None:
        self._write_model(self.page_task_path(task.job_id, task.page_id), task)

    def read_page_task(self, job_id: str, page_id: int) -> PageTask | None:
        return self._read_model(self.page_task_path(job_id, page_id), PageTask)

    def list_page_tasks(self, job_id: str) -> list[PageTask]:
        tasks: list[PageTask] = []
        for path in sorted((self.job_dir(job_id) / "pages").glob("*/task.json")):
            task = self._read_model(path, PageTask)
            if task is not None:
                tasks.append(task)
        return tasks

    def write_page_result(self, result: PageParseResult) -> None:
        revision_path = self.page_result_revision_path(
            result.job_id,
            result.page_id,
            result.result_revision,
        )
        self._write_model(revision_path, result)
        self._write_model(self.page_result_path(result.job_id, result.page_id), result)

    def read_page_result(
        self,
        job_id: str,
        page_id: int,
        *,
        result_revision: int | None = None,
    ) -> PageParseResult | None:
        if result_revision is not None:
            return self._read_model(
                self.page_result_revision_path(job_id, page_id, result_revision),
                PageParseResult,
            )
        return self._read_model(self.page_result_path(job_id, page_id), PageParseResult)

    def _revision_from_result_path(self, path: Path) -> int | None:
        stem = path.stem
        if ".r" not in stem:
            return None
        try:
            return int(stem.split(".r", 1)[1])
        except ValueError:
            return None

    def _best_revision_path(self, page_dir: Path, result_revision: int) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        for path in page_dir.glob("result.r*.json"):
            revision = self._revision_from_result_path(path)
            if revision is None or revision > result_revision:
                continue
            candidates.append((revision, path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def list_page_results(
        self,
        job_id: str,
        *,
        result_revision: int | None = None,
    ) -> list[PageParseResult]:
        results: list[PageParseResult] = []
        page_dirs = sorted((self.job_dir(job_id) / "pages").glob("*"))
        for page_dir in page_dirs:
            if not page_dir.is_dir():
                continue
            if result_revision is None:
                path = page_dir / "result.json"
            else:
                path = self._best_revision_path(page_dir, result_revision)
                if path is None:
                    continue
            if not path.exists():
                continue
            try:
                result = self._read_model(path, PageParseResult)
            except ValidationError:
                continue
            if result is not None:
                results.append(result)
        return results

    def write_chunk_summary(self, summary: ChunkParseSummary) -> None:
        self._write_json(
            self.job_dir(summary.job_id) / "chunks" / f"{summary.chunk_id}.summary.json",
            summary.model_dump(mode="json"),
        )

    def write_final_result(
        self,
        job_id: str,
        result,
        markdown: str,
        text: str,
        *,
        quality_stage: QualityStage = QualityStage.FINAL,
        result_revision: int = 1,
    ) -> None:
        with self.batch():
            self._write_model(
                self.final_stage_revision_json_path(job_id, quality_stage, result_revision),
                result,
            )
            self._write_model(self.final_stage_json_path(job_id, quality_stage), result)
            self._write_model(self.final_json_path(job_id), result)
            self.final_markdown_path(job_id).parent.mkdir(parents=True, exist_ok=True)
            self.final_stage_markdown_path(job_id, quality_stage).write_text(
                markdown,
                encoding="utf-8",
            )
            self.final_stage_text_path(job_id, quality_stage).write_text(
                text,
                encoding="utf-8",
            )
            self.final_markdown_path(job_id).write_text(markdown, encoding="utf-8")
            self.final_text_path(job_id).write_text(text, encoding="utf-8")
            self._commit_if_needed()

    def read_final_result(
        self,
        job_id: str,
        *,
        result_level: ResultLevel = ResultLevel.LATEST,
    ):
        from .types_result import DocumentParseResult

        if result_level == ResultLevel.FINAL:
            path = self.final_stage_json_path(job_id, QualityStage.FINAL)
        elif result_level == ResultLevel.FAST:
            path = self.final_stage_json_path(job_id, QualityStage.FAST)
        else:
            final_path = self.final_stage_json_path(job_id, QualityStage.FINAL)
            fast_path = self.final_stage_json_path(job_id, QualityStage.FAST)
            if final_path.exists():
                path = final_path
            elif fast_path.exists():
                path = fast_path
            else:
                path = self.final_json_path(job_id)
        result = self._read_model(path, DocumentParseResult)
        if result is None:
            raise FileNotFoundError(
                f"No {result_level.value} result available for job_id: {job_id}"
            )
        return result

    def read_result_text(
        self,
        job_id: str,
        format_name: str,
        *,
        result_level: ResultLevel = ResultLevel.LATEST,
    ) -> str:
        if result_level == ResultLevel.FINAL:
            markdown_path = self.final_stage_markdown_path(job_id, QualityStage.FINAL)
            text_path = self.final_stage_text_path(job_id, QualityStage.FINAL)
        elif result_level == ResultLevel.FAST:
            markdown_path = self.final_stage_markdown_path(job_id, QualityStage.FAST)
            text_path = self.final_stage_text_path(job_id, QualityStage.FAST)
        else:
            final_markdown = self.final_stage_markdown_path(job_id, QualityStage.FINAL)
            final_text = self.final_stage_text_path(job_id, QualityStage.FINAL)
            if final_markdown.exists() and final_text.exists():
                markdown_path = final_markdown
                text_path = final_text
            else:
                markdown_path = self.final_stage_markdown_path(job_id, QualityStage.FAST)
                text_path = self.final_stage_text_path(job_id, QualityStage.FAST)
                if not markdown_path.exists() or not text_path.exists():
                    markdown_path = self.final_markdown_path(job_id)
                    text_path = self.final_text_path(job_id)
        if format_name == "markdown":
            return markdown_path.read_text(encoding="utf-8")
        if format_name == "text":
            return text_path.read_text(encoding="utf-8")
        raise ValueError(f"Unsupported text format: {format_name}")

    def get_status(self, job_id: str) -> JobProgressSnapshot | None:
        raw = self.status_store.get(job_id)
        if raw is None:
            return None
        if isinstance(raw, JobProgressSnapshot):
            return raw
        return JobProgressSnapshot.model_validate(raw)

    def set_status(self, job_id: str, snapshot: JobProgressSnapshot) -> None:
        self.status_store.put(job_id, snapshot.model_dump(mode="json"))

    def lookup_idempotency(self, request_fingerprint: str) -> IdempotencyRecord | None:
        raw = self.idempotency_store.get(request_fingerprint)
        if raw is None:
            return None
        if isinstance(raw, IdempotencyRecord):
            return raw
        return IdempotencyRecord.model_validate(raw)

    def store_idempotency(self, record: IdempotencyRecord) -> None:
        self.idempotency_store.put(
            record.request_fingerprint,
            record.model_dump(mode="json"),
            skip_if_exists=False,
        )

    def list_job_ids(self) -> list[str]:
        jobs_dir = self.root_path / "jobs"
        if not jobs_dir.exists():
            return []
        return sorted(path.name for path in jobs_dir.iterdir() if path.is_dir())

    def remove_job(self, job_id: str) -> None:
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        self._commit_if_needed()

    def collect_debug_info(self, job_id: str) -> dict[str, Any]:
        info: dict[str, Any] = {"pages": {}}
        for page_dir in sorted((self.job_dir(job_id) / "pages").glob("*")):
            if not page_dir.is_dir():
                continue
            raw_output = page_dir / "raw_output.txt"
            prompt = page_dir / "prompt.txt"
            page_info: dict[str, Any] = {}
            if raw_output.exists():
                page_info["raw_output_path"] = str(raw_output)
            if prompt.exists():
                page_info["prompt_path"] = str(prompt)
            if page_info:
                info["pages"][page_dir.name] = page_info
        return info

    def extraction_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "extraction"

    def extraction_suggestion_path(self, job_id: str) -> Path:
        return self.extraction_dir(job_id) / "suggestion.json"

    def extraction_result_path(self, job_id: str) -> Path:
        return self.extraction_dir(job_id) / "result.json"

    def extraction_cache_dir(self, job_id: str) -> Path:
        return self.extraction_dir(job_id) / "cache"

    def extraction_suggestion_cache_path(self, job_id: str, fingerprint: str) -> Path:
        return self.extraction_cache_dir(job_id) / f"suggestion.{fingerprint}.json"

    def extraction_result_cache_path(self, job_id: str, fingerprint: str) -> Path:
        return self.extraction_cache_dir(job_id) / f"result.{fingerprint}.json"

    def write_extraction_suggestion(self, job_id: str, suggestion) -> None:
        self._write_model(self.extraction_suggestion_path(job_id), suggestion)

    def read_extraction_suggestion(self, job_id: str):
        from .types_extraction import EntitySuggestionResponse

        return self._read_model(self.extraction_suggestion_path(job_id), EntitySuggestionResponse)

    def write_cached_extraction_suggestion(self, job_id: str, fingerprint: str, suggestion) -> None:
        self._write_model(self.extraction_suggestion_cache_path(job_id, fingerprint), suggestion)

    def read_cached_extraction_suggestion(self, job_id: str, fingerprint: str):
        from .types_extraction import EntitySuggestionResponse

        return self._read_model(
            self.extraction_suggestion_cache_path(job_id, fingerprint),
            EntitySuggestionResponse,
        )

    def write_extraction_result(self, job_id: str, result) -> None:
        self._write_model(self.extraction_result_path(job_id), result)

    def read_extraction_result(self, job_id: str):
        from .types_extraction import EntityExtractionResult

        return self._read_model(self.extraction_result_path(job_id), EntityExtractionResult)

    def write_cached_extraction_result(self, job_id: str, fingerprint: str, result) -> None:
        self._write_model(self.extraction_result_cache_path(job_id, fingerprint), result)

    def read_cached_extraction_result(self, job_id: str, fingerprint: str):
        from .types_extraction import EntityExtractionResult

        return self._read_model(
            self.extraction_result_cache_path(job_id, fingerprint),
            EntityExtractionResult,
        )

    def get_extraction_status(self, job_id: str):
        from .types_extraction import EntityExtractionStatusPayload

        raw = self.status_store.get(f"{job_id}:extraction")
        if raw is None:
            return None
        if isinstance(raw, EntityExtractionStatusPayload):
            return raw
        return EntityExtractionStatusPayload.model_validate(raw)

    def set_extraction_status(self, job_id: str, status_payload) -> None:
        self.status_store.put(f"{job_id}:extraction", status_payload.model_dump(mode="json"))

    def iter_job_manifests(self) -> Iterator[JobManifest]:
        for job_id in self.list_job_ids():
            yield self.read_job_manifest(job_id)
