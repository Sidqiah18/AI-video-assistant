import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI


def get_mistral_model():
    """
    Initializes and returns the Mistral model.

    Returns:
        ChatMistralAI: An instance of the Mistral model.
    """
    mistral_model = os.getenv("MISTRAL_MODEL") or "mistralai/Mistral-7B-Instruct-v0.1"
    return ChatMistralAI(model=mistral_model, temperature=0.3, max_output_tokens=512)


def build_chain(system_prompt: str):
    llm = get_mistral_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting analyst. From the meeting transcript, extract all action items.

For each item, provide:
- Action
- Owner
- Due date
- Priority
- Status
        """
    )
    return chain.invoke(transcript)


def extract_decisions(transcript: str) -> str:
    chain = build_chain(
        """Extract all important questions raised in the meeting.

For each question, provide:
- Question
- Asked by
- Context
- Status
        """
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """Extract all key decisions made during the meeting.

For each decision, provide:
- Decision
- Decided by
- Context
- Impact
        """
    )
    return chain.invoke(transcript)


def _key_decisions(transcript: str) -> str:
    return extract_key_decisions(transcript)
