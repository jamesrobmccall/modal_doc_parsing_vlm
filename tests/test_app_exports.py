from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from modal_doc_parsing_vlm.config import ENABLED_RUNTIME_PROFILES


def test_fallback_engine_class_exports_match_enabled_profiles():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = spec_from_file_location("modal_app", app_path)
    assert spec is not None and spec.loader is not None
    app = module_from_spec(spec)
    spec.loader.exec_module(app)

    for profile in ENABLED_RUNTIME_PROFILES:
        class_name = f"{profile.title()}FallbackEngine"
        assert hasattr(app, class_name), f"Missing app export: {class_name}"
