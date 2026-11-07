import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI


def get_llm():
    """Create and return the chat model for the RAG pipeline with automatic retries."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set.")

    model_name = os.getenv("MISTRAL_MODEL") or "ministral-8b-latest"
    return ChatMistralAI(model=model_name, temperature=0.2, max_tokens=512, max_retries=5)


def format_docs(docs) -> str:
    """Convert retrieved documents into a single context string."""
    if not docs:
        return ""

    return "\n\n---\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


def build_rag_chain(transcript: str = ""):
    """Build a retrieval-augmented QA chain that answers strictly from transcript chunks."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant answering questions about this specific video/audio transcript.\n"
        "Instructions:\n"
        "1. Base your answer strictly on the transcript context provided below.\n"
        "2. If the user's message is a greeting or polite remark (e.g. 'hello', 'thank you', 'okay'), respond warmly and politely without dumping the context.\n"
        "3. If the user's question asks about something that is NOT mentioned or explained in the context, clearly answer: "
        "'That is not mentioned in this video.' Do NOT invent facts or borrow examples from outside topics.\n"
        "4. Keep your answer concise, accurate, and direct.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
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
