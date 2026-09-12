import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_audio
from utils.core.rag_engine import load_rag_chain
from utils.core.transcriber import transcribe_all
from utils.core.vector_store import add_transcript, query_transcript
from utils.extractor import extract_action_items, extract_key_decisions
from utils.summarizer import generate_title, summarize_transcript

load_dotenv()


st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")


def transcribe_source(source: str, translate: bool = False):
    chunks = process_audio(source)
    transcript = transcribe_all(chunks, translate=translate)
    return transcript


def process_and_index_transcript(transcript: str, source_name: str = "uploaded_audio"):
    title = generate_title(transcript)
    summary = summarize_transcript(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)

    add_transcript(transcript, metadata={"title": title, "source": source_name})

    return {
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "transcript": transcript,
    }


st.title("🎬 AI Video Assistant")
st.caption("Upload audio/video or paste a transcript, then ask questions from the indexed content.")

with st.sidebar:
    st.header("Source")
    source_type = st.radio("Choose input type", ["Upload audio/video", "YouTube URL", "Paste transcript"])

    transcript_text = ""
    source_value = ""

    if source_type == "Upload audio/video":
        uploaded_file = st.file_uploader("Upload a file", type=["mp3", "wav", "m4a", "mp4", "webm", "mov"])
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                source_value = tmp.name

    elif source_type == "YouTube URL":
        source_value = st.text_input("YouTube link")

    else:
        transcript_text = st.text_area("Paste transcript", height=250)

    translate = st.checkbox("Translate transcript to English")
    run_button = st.button("Process transcript")

if run_button:
    if source_type == "Paste transcript":
        if not transcript_text.strip():
            st.warning("Please paste a transcript first.")
        else:
            result = process_and_index_transcript(transcript_text, source_name="pasted_transcript")
            st.session_state["result"] = result
            st.success("Transcript indexed successfully.")
    elif source_value:
        try:
            with st.spinner("Transcribing and indexing content..."):
                transcript = transcribe_source(source_value, translate=translate)
                result = process_and_index_transcript(transcript, source_name=source_value)
                st.session_state["result"] = result
            st.success("Transcription complete and indexed.")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
    else:
        st.warning("Please provide a valid input first.")

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader(result["title"])

    tabs = st.tabs(["Summary", "Transcript", "Action Items", "Decisions", "Ask Question"])

    with tabs[0]:
        st.write(result["summary"])

    with tabs[1]:
        st.text_area("Full transcript", result["transcript"], height=250)

    with tabs[2]:
        st.write(result["action_items"])

    with tabs[3]:
        st.write(result["decisions"])

    with tabs[4]:
        question = st.text_input("Ask a question about the transcript")
        if question:
            docs = query_transcript(question, k=4)
            if not docs:
                st.warning("No matching transcript chunks were found.")
            else:
                chain = load_rag_chain()
                answer = chain.invoke({"documents": docs, "question": question})
                st.write(answer)
