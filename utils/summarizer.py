import os

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
    mistral_model = os.getenv("MISTRAL_MODEL") or "mistralai/Mistral-7B-Instruct-v0.1"
    return ChatMistralAI(model=mistral_model, temperature=0.2, max_output_tokens=512)


def split_transcript(transcript: str) -> list[str]:
    """
    Splits a transcript into smaller chunks for processing.

    Args:
        transcript (str): The full transcript to be split.

    Returns:
        list[str]: A list of transcript chunks.
    """
    chunk_size = 1000
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(transcript)


def summarize_transcript(transcript: str) -> str:
    """
    Summarizes a transcript using the Mistral model.

    Args:
        transcript (str): The full transcript to be summarized.

    Returns:
        str: The summarized text.
    """
    mistral_model = get_mistral_model()
    prompt_template = ChatPromptTemplate.from_template(
        "Summarize the following transcript:\n\n{transcript}"
    )
    output_parser = StrOutputParser()

    summarize_chain = prompt_template | mistral_model | output_parser
    chunks = split_transcript(transcript)
    summary = [summarize_chain.invoke({"transcript": chunk}) for chunk in chunks]
    concatenated_summary = " ".join(summary)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    combined_chain = RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | mistral_model | output_parser
    return combined_chain.invoke(concatenated_summary)


def generate_title(transcript: str) -> str:
    llm = get_mistral_model()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title (max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    return title_chain.invoke({"text": transcript}).strip()
