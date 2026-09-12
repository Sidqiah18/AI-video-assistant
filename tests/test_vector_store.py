import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'utils' / 'core' / 'vector_store.py'

spec = importlib.util.spec_from_file_location('vector_store_module', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rag_api_exists():
    assert hasattr(module, 'add_transcript')
    assert hasattr(module, 'query_transcript')
    assert hasattr(module, 'rag_answer')


def test_split_text_returns_chunks():
    chunks = module.split_transcript('Hello world. ' * 200)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
