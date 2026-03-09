from __future__ import annotations

import json
import threading
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_doc_parsing_vlm.extraction_client import build_job_extraction_session_id
from modal_doc_parsing_vlm.types_extraction import (
    EntityDefinition,
    EntityFieldDefinition,
    ExtractionFieldType,
)


def _load_app_module():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = spec_from_file_location("modal_app_extraction_reuse", app_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entity(name: str) -> EntityDefinition:
    return EntityDefinition(
        entity_name=name,
        description=f"{name} fields",
        fields=[
            EntityFieldDefinition(
                name="value",
                field_type=ExtractionFieldType.STRING,
                description="Extracted value",
                required=False,
            )
        ],
    )


def _payload_identity(payload: dict[str, object]) -> tuple[str, str]:
    user_content = str(payload["messages"][1]["content"])
    entity_name = user_content.split("Entity: ", 1)[1].splitlines()[0].strip()
    page_token = "PAGE-1" if "PAGE-1" in user_content else "PAGE-2"
    return entity_name, page_token


def test_online_per_page_extraction_reuses_one_job_session_and_preserves_order(monkeypatch):
    app_module = _load_app_module()
    monkeypatch.setattr(
        app_module,
        "_ensure_extraction_server_ready",
        lambda *, timeout_seconds: "https://extract.example",
    )
    monkeypatch.setattr(app_module, "EXTRACTION_TARGET_INPUTS", 4)
    monkeypatch.setattr(app_module, "EXTRACTION_MAX_RUNNING_REQUESTS", 2)

    status_updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        app_module,
        "_set_extraction_status",
        lambda _storage, **kwargs: status_updates.append(kwargs),
    )

    lock = threading.Lock()
    active_requests = 0
    max_active_requests = 0
    session_ids: list[str] = []
    delays = {
        ("Invoice", "PAGE-1"): 0.05,
        ("Invoice", "PAGE-2"): 0.01,
        ("Vendor", "PAGE-1"): 0.03,
        ("Vendor", "PAGE-2"): 0.02,
    }

    def fake_call(payload, *, session_id, base_url=None):
        nonlocal active_requests, max_active_requests
        entity_name, page_token = _payload_identity(payload)
        with lock:
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            session_ids.append(session_id)
        time.sleep(delays[(entity_name, page_token)])
        with lock:
            active_requests -= 1
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"marker": f"{entity_name}:{page_token}", "base_url": base_url}
                        )
                    }
                }
            ]
        }, 7

    monkeypatch.setattr(app_module, "_call_extraction_chat_completion", fake_call)

    request = SimpleNamespace(entities=[_entity("Invoice"), _entity("Vendor")])
    results, total_inference_ms = app_module._run_per_page_extraction_via_online_server(
        job_id="job-1",
        request=request,
        page_tasks=[(1, "PAGE-1"), (2, "PAGE-2")],
        storage=object(),
    )

    assert max_active_requests == 2
    assert set(session_ids) == {build_job_extraction_session_id("job-1")}
    assert [(result.entity_name, result.page_id, result.data["marker"]) for result in results] == [
        ("Invoice", 1, "Invoice:PAGE-1"),
        ("Invoice", 2, "Invoice:PAGE-2"),
        ("Vendor", 1, "Vendor:PAGE-1"),
        ("Vendor", 2, "Vendor:PAGE-2"),
    ]
    assert total_inference_ms == 28
    assert [update["requests_completed"] for update in status_updates] == [1, 2, 3, 4]
    assert status_updates[-1]["pages_processed"] == 2


def test_online_per_page_extraction_propagates_failures(monkeypatch):
    app_module = _load_app_module()
    monkeypatch.setattr(
        app_module,
        "_ensure_extraction_server_ready",
        lambda *, timeout_seconds: "https://extract.example",
    )
    monkeypatch.setattr(app_module, "EXTRACTION_TARGET_INPUTS", 3)
    monkeypatch.setattr(app_module, "EXTRACTION_MAX_RUNNING_REQUESTS", 2)

    status_updates: list[dict[str, object]] = []
    monkeypatch.setattr(
        app_module,
        "_set_extraction_status",
        lambda _storage, **kwargs: status_updates.append(kwargs),
    )

    def fake_call(payload, *, session_id, base_url=None):
        entity_name, page_token = _payload_identity(payload)
        if (entity_name, page_token) == ("Invoice", "PAGE-2"):
            raise RuntimeError("boom")
        return {
            "choices": [{"message": {"content": json.dumps({"marker": f"{entity_name}:{page_token}"})}}]
        }, 5

    monkeypatch.setattr(app_module, "_call_extraction_chat_completion", fake_call)

    request = SimpleNamespace(entities=[_entity("Invoice"), _entity("Vendor")])
    with pytest.raises(RuntimeError, match="boom"):
        app_module._run_per_page_extraction_via_online_server(
            job_id="job-2",
            request=request,
            page_tasks=[(1, "PAGE-1"), (2, "PAGE-2")],
            storage=object(),
        )

    assert len(status_updates) < 4
