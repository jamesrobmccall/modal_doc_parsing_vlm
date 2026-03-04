from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import MIME_TO_SUFFIX, PARSER_VERSION
from .types_api import SubmitDocumentParseRequest


@dataclass(frozen=True)
class ResolvedSource:
    data: bytes
    file_name: str
    content_hash: str


def _guess_file_name_from_url(url: str, mime_type: str) -> str:
    parsed = urlparse(url)
    basename = Path(parsed.path).name or "document"
    suffix = Path(basename).suffix
    if suffix:
        return basename
    return f"{basename}{MIME_TO_SUFFIX[mime_type]}"


def _decode_base64_payload(payload: str) -> bytes:
    return base64.b64decode(payload.encode("utf-8"), validate=True)


def resolve_source_bytes(
    request: SubmitDocumentParseRequest,
    storage,
) -> ResolvedSource:
    source = request.source
    if source.type == "bytes":
        data = _decode_base64_payload(source.base64)
        file_name = f"inline{MIME_TO_SUFFIX[request.mime_type.value]}"
    elif source.type == "upload_ref":
        upload = storage.read_upload(source.upload_id)
        if upload is None:
            raise FileNotFoundError(f"Unknown upload_id: {source.upload_id}")
        data = upload["data"]
        file_name = upload["file_name"]
    else:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            response = client.get(source.url)
            response.raise_for_status()
            data = response.content
        file_name = _guess_file_name_from_url(source.url, request.mime_type.value)

    content_hash = hashlib.sha256(data).hexdigest()
    return ResolvedSource(data=data, file_name=file_name, content_hash=content_hash)


def compute_request_fingerprint(
    request: SubmitDocumentParseRequest,
    content_hash: str,
    *,
    parser_version: str = PARSER_VERSION,
) -> str:
    normalized = request.model_dump(mode="json")
    normalized["source"] = {"type": "content_hash", "sha256": content_hash}
    normalized["parser_version"] = parser_version
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
