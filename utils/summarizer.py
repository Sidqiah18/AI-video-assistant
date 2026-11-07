import os
import re
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def get_mistral_model():
    """
    Initializes and returns the Mistral model with built-in retries.

    Returns:
        ChatMistralAI: An instance of the Mistral model.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    mistral_model = os.getenv("MISTRAL_MODEL")
    try:
        import streamlit as st
        if not api_key and "MISTRAL_API_KEY" in st.secrets:
            api_key = st.secrets["MISTRAL_API_KEY"]
        if not mistral_model and "MISTRAL_MODEL" in st.secrets:
            mistral_model = st.secrets["MISTRAL_MODEL"]
    except Exception:
        pass

    mistral_model = mistral_model or "ministral-8b-latest"
    kwargs = {
        "model": mistral_model,
        "temperature": 0.2,
        "max_tokens": 512,
        "max_retries": 5,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return ChatMistralAI(**kwargs)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=15),
    reraise=True,
)
def invoke_chain_with_backoff(chain, input_data):
    """Invoke a LangChain runnable with automatic exponential backoff on rate limits."""
    return chain.invoke(input_data)


def clean_llm_output(text: str) -> str:
    """Strip common conversational introductory and concluding lines from LLM responses."""
    lines = text.strip().splitlines()
    while lines and (
        not lines[0].strip()
        or re.match(
            r"^(here('s| is)|below is|sure|certainly|this is|in this summary|the following is)[\w\s,.:\-]*:?\s*$",
            lines[0].strip(),
            re.IGNORECASE,
        )
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


def split_transcript(transcript: str) -> list[str]:
    """
    Splits a transcript into manageable chunks for processing.
    Uses large chunks (12,000 chars) to take advantage of modern 128k context windows
    and minimize API calls against rate limits.
    """
    chunk_size = 12000
    chunk_overlap = 800
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(transcript)


def summarize_transcript(transcript: str, is_meeting: bool = False) -> str:
    """
    Summarizes a transcript using the Mistral model, adapting tone and structure
    depending on whether the content is an interactive meeting or a video/tutorial.

    Args:
        transcript (str): The full transcript to be summarized.
        is_meeting (bool): True if the transcript is a team meeting; False for general videos/tutorials.

    Returns:
        str: The summarized text in Markdown format.
    """
    mistral_model = get_mistral_model()
    prompt_template = ChatPromptTemplate.from_template(
        "Summarize the key points of the following transcript segment clearly and concisely. "
        "Do not include conversational filler:\n\n{transcript}"
    )
    output_parser = StrOutputParser()

    summarize_chain = prompt_template | mistral_model | output_parser
    chunks = split_transcript(transcript)
    if not chunks:
        return "No content to summarize."

    # If transcript fits in one large chunk, summarize directly in 1 call!
    if len(chunks) == 1:
        concatenated_summary = chunks[0]
    else:
        summary = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                time.sleep(1.2)  # Respect Mistral free tier RPS rate limit
            summary.append(invoke_chain_with_backoff(summarize_chain, {"transcript": chunk}))
        concatenated_summary = " ".join(summary)

    if is_meeting:
        system_prompt = (
            "You are an expert meeting analyst. Combine these notes into a structured, professional meeting summary in bullet points covering:\n"
            "- Meeting Objectives\n"
            "- Main Discussion Points\n"
            "- Key Outcomes\n\n"
            "CRITICAL: Do NOT include conversational preambles (e.g., 'Here is a summary...'). "
            "Start directly with the markdown summary."
        )
    else:
        system_prompt = (
            "You are an expert content summarizer. Combine these notes into a structured, clear video summary in bullet points covering:\n"
            "- Overview & Core Topic\n"
            "- Key Concepts & Insights\n"
            "- Practical Takeaways\n\n"
            "CRITICAL: This is an educational video, talk, or tutorial, NOT a business meeting. "
            "Do NOT use the word 'meeting' or corporate meeting terminology. "
            "Do NOT include conversational preambles (e.g., 'Here is a summary...'). "
            "Start directly with the markdown summary."
        )

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | mistral_model
        | output_parser
    )
    raw_summary = invoke_chain_with_backoff(combined_chain, concatenated_summary)
    return clean_llm_output(raw_summary)


def generate_title(transcript: str) -> str:
    """Generate a clean, concise title (max 8 words) for the audio/video with rate limit retries."""
    llm = get_mistral_model()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the transcript excerpt, generate a concise, professional title (max 8 words) describing the topic. "
                "Do NOT use the word 'meeting' unless it is clearly a meeting. "
                "Only return the title text, nothing else. Do not use quotes or introductory words.",
            ),
            ("human", "{text}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    sample = transcript[:3000] if transcript else ""
    raw_title = invoke_chain_with_backoff(title_chain, {"text": sample}).strip()
    clean_title = re.sub(r'^["\'\s*#]+|["\'\s*#]+$', '', raw_title)
    return clean_title or "Audio / Video Content"
