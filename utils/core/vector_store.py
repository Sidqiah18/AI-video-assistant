import os
from typing import Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vector_db")
DEFAULT_COLLECTION = "transcripts"

os.makedirs(CHROMA_DIR, exist_ok=True)


def split_transcript(transcript: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Split transcript text into manageable chunks for vector search."""
    if not transcript or not transcript.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(transcript)


def get_embedding_model() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vector_store(collection_name: str = DEFAULT_COLLECTION, persist_directory: str = CHROMA_DIR):
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=persist_directory,
    )


def add_transcript(transcript: str, metadata: Optional[dict[str, Any]] = None, source: Optional[str] = None):
    """Embed and save transcript chunks in Chroma."""
    if not transcript or not transcript.strip():
        return []

    doc_metadata = dict(metadata or {})
    if source:
        doc_metadata.setdefault("source", source)

    chunks = split_transcript(transcript)
    documents = [
        Document(page_content=chunk, metadata={**doc_metadata, "chunk_index": idx})
        for idx, chunk in enumerate(chunks)
    ]

    if not documents:
        return []

    vector_store = get_vector_store()
    return vector_store.add_documents(documents)


def query_transcript(question: str, k: int = 4):
    """Retrieve the most relevant transcript chunks for a question."""
    if not question or not question.strip():
        return []

    vector_store = get_vector_store()
    return vector_store.similarity_search(question, k=k)


def rag_answer(question: str, transcript: Optional[str] = None, k: int = 4) -> str:
    """Answer a question from transcript content using retrieval-augmented generation."""
    if transcript:
        add_transcript(transcript, source="direct_input")
        context_docs = [
            Document(page_content=chunk, metadata={"source": "direct_input"})
            for chunk in split_transcript(transcript)[:k]
        ]
    else:
        context_docs = query_transcript(question, k=k)

    if not context_docs:
        return "No transcript data has been indexed yet. Add a transcript before asking a question."

    context_text = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        try:
            llm = ChatMistralAI(
                model=os.getenv("MISTRAL_MODEL") or "mistral-large-latest",
                temperature=0.2,
                max_tokens=512,
            )
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user's question using only the transcript context below. "
                "If the answer is not present in the context, say you do not know.\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
            return (prompt | llm | StrOutputParser()).invoke({"context": context_text, "question": question})
        except Exception:
            pass

    return (
        "Based on the transcript context:\n\n"
        + "\n\n".join(f"- {doc.page_content[:500]}" for doc in context_docs[:3])
    )


__all__ = [
    "CHROMA_DIR",
    "DEFAULT_COLLECTION",
    "split_transcript",
    "get_embedding_model",
    "get_vector_store",
    "add_transcript",
    "query_transcript",
    "rag_answer",
]


