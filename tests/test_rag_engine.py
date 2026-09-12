import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'utils' / 'core' / 'rag_engine.py'

spec = importlib.util.spec_from_file_location('rag_engine_module', MODULE_PATH)
module = importlib.util.module_from_spec(spec)


def test_rag_engine_exports_expected_functions():
    spec.loader.exec_module(module)
    assert hasattr(module, 'get_llm')
    assert hasattr(module, 'format_docs')
    assert hasattr(module, 'build_rag_chain')
    assert hasattr(module, 'load_rag_chain')


def test_format_docs_joins_content_from_documents():
    spec.loader.exec_module(module)
    docs = [
        type('Doc', (), {'page_content': 'first paragraph'})(),
        type('Doc', (), {'page_content': 'second paragraph'})(),
    ]
    result = module.format_docs(docs)
    assert 'first paragraph' in result
    assert 'second paragraph' in result


def test_load_rag_chain_returns_chain():
    spec.loader.exec_module(module)
    chain = module.load_rag_chain()
    assert hasattr(chain, 'invoke')
