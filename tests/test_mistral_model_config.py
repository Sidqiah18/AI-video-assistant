import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module_name", ["summarizer", "extractor"])
def test_mistral_model_uses_supported_default(monkeypatch, module_name):
    module_path = PROJECT_ROOT / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"{module_name}_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    created_models = []

    def fake_chat_mistral_ai(**kwargs):
        created_models.append(kwargs)
        return object()

    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    monkeypatch.setattr(module, "ChatMistralAI", fake_chat_mistral_ai)

    module.get_mistral_model()

    assert created_models[0]["model"] == "ministral-8b-latest"


@pytest.mark.parametrize("module_name", ["summarizer", "extractor"])
def test_mistral_model_respects_environment_override(monkeypatch, module_name):
    module_path = PROJECT_ROOT / "utils" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"{module_name}_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    created_models = []

    def fake_chat_mistral_ai(**kwargs):
        created_models.append(kwargs)
        return object()

    monkeypatch.setenv("MISTRAL_MODEL", "custom-model")
    monkeypatch.setattr(module, "ChatMistralAI", fake_chat_mistral_ai)

    module.get_mistral_model()

    assert created_models[0]["model"] == "custom-model"
