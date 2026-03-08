from modal_doc_parsing_vlm import engine_extraction


class FakeImage:
    def __init__(self) -> None:
        self.env_values: list[dict[str, str]] = []
        self.run_function_calls: list[tuple[object, dict]] = []
        self.local_sources: list[str] = []

    def entrypoint(self, _value):
        return self

    def uv_pip_install(self, *_args, **_kwargs):
        return self

    def env(self, values):
        self.env_values.append(values)
        return self

    def run_function(self, fn, **kwargs):
        self.run_function_calls.append((fn, kwargs))
        return self

    def add_local_python_source(self, source):
        self.local_sources.append(source)
        return self


class FakeImageFactory:
    last_image: FakeImage | None = None

    @staticmethod
    def from_registry(*_args, **_kwargs):
        FakeImageFactory.last_image = FakeImage()
        return FakeImageFactory.last_image


def test_build_extraction_image_skips_deepgemm_when_disabled(monkeypatch):
    monkeypatch.setattr(engine_extraction.modal, "Image", FakeImageFactory)
    monkeypatch.setattr(engine_extraction, "EXTRACTION_ENABLE_DEEPGEMM", False)

    image = engine_extraction._build_extraction_image(object(), object())

    assert image is FakeImageFactory.last_image
    assert image is not None
    assert image.run_function_calls == []
    assert image.local_sources == ["modal_doc_parsing_vlm"]
    assert "SGLANG_ENABLE_JIT_DEEPGEMM" not in image.env_values[-1]


def test_build_extraction_image_compiles_deepgemm_when_enabled(monkeypatch):
    monkeypatch.setattr(engine_extraction.modal, "Image", FakeImageFactory)
    monkeypatch.setattr(engine_extraction, "EXTRACTION_ENABLE_DEEPGEMM", True)
    monkeypatch.setattr(engine_extraction, "EXTRACTION_GPU", "H100:1")

    image = engine_extraction._build_extraction_image("hf-volume", "deepgemm-volume")

    assert image is FakeImageFactory.last_image
    assert image is not None
    assert len(image.run_function_calls) == 1
    _fn, kwargs = image.run_function_calls[0]
    assert kwargs["gpu"] == "H100:1"
    assert kwargs["volumes"][str(engine_extraction.HF_CACHE_ROOT)] == "hf-volume"
    assert kwargs["volumes"][str(engine_extraction.DEEPGEMM_CACHE_ROOT)] == "deepgemm-volume"
    assert image.env_values[-1]["SGLANG_ENABLE_JIT_DEEPGEMM"] == "1"
