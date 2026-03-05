import pytest
from pydantic import ValidationError

from modal_doc_parsing_vlm.types_api import SubmitDocumentParseRequest
from modal_doc_parsing_vlm.types_result import LatencyProfile, OutputFormat, ResultLevel


def test_submit_request_adds_json_output():
    request = SubmitDocumentParseRequest.model_validate(
        {
            "source": {"type": "bytes", "base64": "aGVsbG8="},
            "mime_type": "image/png",
            "mode": "balanced",
            "output_formats": ["markdown"],
        }
    )

    assert request.output_formats == [OutputFormat.MARKDOWN, OutputFormat.JSON]


def test_submit_request_rejects_empty_output_formats():
    with pytest.raises(ValidationError):
        SubmitDocumentParseRequest.model_validate(
            {
                "source": {"type": "bytes", "base64": "aGVsbG8="},
                "mime_type": "image/png",
                "output_formats": [],
            }
        )


def test_submit_request_rejects_invalid_page_range():
    with pytest.raises(ValidationError):
        SubmitDocumentParseRequest.model_validate(
            {
                "source": {"type": "bytes", "base64": "aGVsbG8="},
                "mime_type": "application/pdf",
                "output_formats": ["json"],
                "page_range": "1-a",
            }
        )


def test_submit_request_rejects_invalid_mime_type():
    with pytest.raises(ValidationError):
        SubmitDocumentParseRequest.model_validate(
            {
                "source": {"type": "bytes", "base64": "aGVsbG8="},
                "mime_type": "application/octet-stream",
                "output_formats": ["json"],
            }
        )


def test_submit_request_defaults_result_level_and_latency_profile():
    request = SubmitDocumentParseRequest.model_validate(
        {
            "source": {"type": "bytes", "base64": "aGVsbG8="},
            "mime_type": "image/png",
            "output_formats": ["json"],
        }
    )
    assert request.result_level == ResultLevel.LATEST
    assert request.latency_profile == LatencyProfile.BALANCED
