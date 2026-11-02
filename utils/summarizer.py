import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_mistral_model():
    """
    Initializes and returns the Mistral model.

    Returns:
        ChatMistralAI: An instance of the Mistral model.
    """
    mistral_model = os.getenv("MISTRAL_MODEL") or "ministral-8b-latest"
    return ChatMistralAI(model=mistral_model, temperature=0.2, max_tokens=512)


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
    Splits a transcript into smaller chunks for processing.

    Args:
        transcript (str): The full transcript to be split.

    Returns:
        list[str]: A list of transcript chunks.
    """
    chunk_size = 1200
    chunk_overlap = 150
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

    # If transcript is short enough, summarize directly to save latency and preserve context
    if len(chunks) == 1:
        concatenated_summary = chunks[0]
    else:
        summary = [summarize_chain.invoke({"transcript": chunk}) for chunk in chunks]
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
    raw_summary = combined_chain.invoke(concatenated_summary)
    return clean_llm_output(raw_summary)


def generate_title(transcript: str) -> str:
    """Generate a clean, concise title (max 8 words) for the audio/video."""
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
    raw_title = title_chain.invoke({"text": sample}).strip()
    clean_title = re.sub(r'^["\'\s*#]+|["\'\s*#]+$', '', raw_title)
    return clean_title or "Audio / Video Content"
