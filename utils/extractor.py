import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def get_mistral_model():
    """
    Initializes and returns the Mistral model.

    Returns:
        ChatMistralAI: An instance of the Mistral model with automatic retries.
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


def build_chain(system_prompt: str):
    llm = get_mistral_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | prompt | llm | StrOutputParser()


def is_meeting_transcript(transcript: str) -> bool:
    """
    Determines whether the transcript represents a team/business meeting
    (with colleagues discussing project tasks, assignments, status updates, or decisions)
    versus a general video (such as a YouTube tutorial, tech talk, lecture, podcast, or monologue).
    """
    if not transcript or not transcript.strip():
        return False

    llm = get_mistral_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an audio classifier. Analyze the excerpt and determine if it represents an interactive business or team meeting "
                "(with colleagues discussing agenda items, assignments, or decisions) OR if it is a general video "
                "(such as an educational YouTube tutorial, tech talk, lecture, explainer, or presentation).\n"
                "Answer strictly with one word: 'MEETING' or 'GENERAL'."
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        sample = transcript[:3000]
        classification = invoke_chain_with_backoff(chain, {"text": sample}).strip().upper()
        return "MEETING" in classification
    except Exception:
        return False


def extract_action_items(transcript: str) -> str:
    """Extract action items from a meeting transcript."""
    chain = build_chain(
        """You are an expert meeting analyst. From the meeting transcript, extract all action items.
Do NOT include conversational preambles. Start directly with the items.

For each item, provide:
- Action
- Owner
- Due date
- Priority
- Status
        """
    )
    sample = transcript[:15000] if transcript else ""
    return clean_llm_output(invoke_chain_with_backoff(chain, sample))


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions from a meeting transcript."""
    chain = build_chain(
        """Extract all key decisions made during the meeting.
Do NOT include conversational preambles. Start directly with the decisions.

For each decision, provide:
- Decision
- Decided by
- Context
- Impact
        """
    )
    sample = transcript[:15000] if transcript else ""
    return clean_llm_output(invoke_chain_with_backoff(chain, sample))


def extract_key_takeaways(transcript: str) -> str:
    """
    Extract core concepts, mechanisms, techniques, and practical takeaways
    from a tutorial, educational video, or presentation transcript.
    """
    chain = build_chain(
        """You are an expert educational content analyst. Extract the primary key takeaways, core concepts,
and techniques explained in this video/tutorial transcript.
Organize them with clear bullet points and bold section headers.
CRITICAL: Do NOT use meeting terminology (e.g. owners, deciders, agenda).
Do NOT include any conversational filler (e.g., 'Here are the key takeaways...').
Start directly with the markdown bullet points.
        """
    )
    sample = transcript[:15000] if transcript else ""
    return clean_llm_output(invoke_chain_with_backoff(chain, sample))


def extract_decisions(transcript: str) -> str:
    """Kept for backward compatibility."""
    chain = build_chain(
        """Extract all important questions raised in the session.
For each question, provide:
- Question
- Asked by / Topic
- Context
- Status
        """
    )
    sample = transcript[:15000] if transcript else ""
    return clean_llm_output(invoke_chain_with_backoff(chain, sample))


def _key_decisions(transcript: str) -> str:
    return extract_key_decisions(transcript)
