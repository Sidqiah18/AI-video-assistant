import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI


def get_llm():
    """Create and return the chat model for the RAG pipeline."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set.")

    model_name = os.getenv("MISTRAL_MODEL") or "mistral-large-latest"
    return ChatMistralAI(model=model_name, temperature=0.2, max_tokens=512)


def format_docs(docs) -> str:
    """Convert retrieved documents into a single context string."""
    if not docs:
        return ""

    return "\n\n---\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


def build_rag_chain(transcript: str = ""):
    """Build a retrieval-augmented QA chain that answers from transcript chunks."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer using the transcript context only. "
        "If the answer is not in the context, say you do not know.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}"
    )

    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"context": format_docs(x["documents"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def load_rag_chain():
    """Load the configured RAG chain for transcript question answering."""
    return build_rag_chain()


__all__ = ["get_llm", "format_docs", "build_rag_chain", "load_rag_chain"]
