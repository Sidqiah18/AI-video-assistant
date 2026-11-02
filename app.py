import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Fix Windows console charmap encoding issues with unicode in prints/logs
for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_audio
from utils.core.rag_engine import load_rag_chain
from utils.core.transcriber import transcribe_all
from utils.core.vector_store import add_transcript, clear_vector_store, query_transcript
from utils.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_key_takeaways,
    is_meeting_transcript,
)
from utils.summarizer import generate_title, summarize_transcript

load_dotenv()


st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")


def handle_conversational_query(query: str) -> Optional[str]:
    """Return an immediate natural polite reply for greetings, thanks, and simple acknowledgments."""
    cleaned = re.sub(r"[^\w\s]", "", query.strip().lower())
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy", "sup"}
    thanks = {
        "thank you", "thanks", "thx", "ty", "okay thank you", "ok thank you",
        "great thanks", "great thank you", "thank you so much", "thanks a lot",
        "thank u", "many thanks", "thanks for help", "thanks for the help"
    }
    acks = {"ok", "okay", "cool", "got it", "understood", "awesome", "perfect", "great", "nice", "bye", "goodbye", "see ya"}

    if cleaned in thanks:
        return "You're very welcome! Feel free to ask anything else about the video."
    if cleaned in greetings:
        return "Hello! What questions can I answer for you about this video?"
    if cleaned in acks:
        return "Sounds good! Let me know if you need any other insights or details."
    return None


def process_and_index_transcript(transcript: str, source_name: str = "uploaded_audio", mode: str = "Auto-detect"):
    if mode == "Meeting Minutes":
        is_meeting = True
    elif mode == "Video / Tutorial":
        is_meeting = False
    else:
        is_meeting = is_meeting_transcript(transcript)

    title = generate_title(transcript)
    summary = summarize_transcript(transcript, is_meeting=is_meeting)

    if is_meeting:
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        takeaways = None
    else:
        action_items = None
        decisions = None
        takeaways = extract_key_takeaways(transcript)

    # Embed and save, clearing any previous video's chunks to prevent cross-video contamination
    add_transcript(
        transcript,
        metadata={"title": title, "source": source_name, "is_meeting": str(is_meeting)},
        clear_existing=True,
    )

    return {
        "title": title,
        "is_meeting": is_meeting,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "takeaways": takeaways,
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
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("Sample Tutorial"):
                st.session_state["sample_text"] = (
                    "Welcome back to another tutorial! Today we are exploring Retrieval-Augmented Generation, or RAG.\n"
                    "Large Language Models like GPT and Mistral are impressive at general language tasks, but they suffer from hallucinations.\n"
                    "Hallucination means the model confidently generates incorrect or outdated facts.\n"
                    "RAG solves this by introducing an external knowledge retrieval step. Before the model answers your question, "
                    "a vector database like ChromaDB searches for relevant paragraphs from your company docs.\n"
                    "These retrieved chunks are then passed into the prompt as grounding context, preventing hallucinations and guaranteeing up-to-date answers."
                )
        with sample_col2:
            if st.button("Sample Meeting"):
                st.session_state["sample_text"] = (
                    "Alex: Good morning team, let's review the launch plan for Project Falcon.\n"
                    "Sarah: The backend services are 100% deployed. However, load testing revealed latency spikes on high concurrency.\n"
                    "Michael: I will optimize the Redis caching layer by Friday to bring latency down.\n"
                    "Alex: Agreed. Michael owns the Redis optimization due Friday with high priority.\n"
                    "Sarah: We decided to postpone the public beta from Monday to next Wednesday to verify the cache fixes.\n"
                    "Alex: Confirmed decision: Beta launch moved to next Wednesday."
                )

        default_text = st.session_state.get("sample_text", "")
        transcript_text = st.text_area("Paste transcript", value=default_text, height=220)

    st.markdown("---")
    content_mode = st.selectbox(
        "Content format",
        ["Auto-detect", "Video / Tutorial", "Meeting Minutes"],
        help="Auto-detect dynamically adjusts whether to extract key concepts or meeting minutes (action items/decisions).",
    )
    translate = st.checkbox("Translate transcript to English")
    run_button = st.button("Process transcript", type="primary")

if run_button:
    # Reset chat history for the new video
    st.session_state["qa_history"] = []

    if source_type == "Paste transcript":
        if not transcript_text.strip():
            st.warning("Please paste a transcript first.")
        else:
            with st.spinner("Analyzing transcript and indexing into vector database..."):
                result = process_and_index_transcript(
                    transcript_text,
                    source_name="pasted_transcript",
                    mode=content_mode,
                )
                st.session_state["result"] = result
            st.toast("Transcript processed and indexed!", icon="✅")
    elif source_value:
        try:
            with st.status("Processing audio...", expanded=True) as status:
                st.write("1️⃣ Downloading & converting audio...")
                chunks = process_audio(source_value)
                st.write(f"2️⃣ Transcribing {len(chunks)} audio chunk(s) using Whisper...")
                transcript = transcribe_all(chunks, translate=translate)
                st.write("3️⃣ Generating title, summary & indexing into ChromaDB...")
                result = process_and_index_transcript(
                    transcript,
                    source_name=source_value,
                    mode=content_mode,
                )
                st.session_state["result"] = result
                status.update(label="Complete! Results ready below.", state="complete", expanded=False)
            st.toast("Transcription complete and indexed!", icon="✅")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
    else:
        st.warning("Please provide a valid input first.")

if "result" in st.session_state:
    result = st.session_state["result"]
    is_meeting = result.get("is_meeting", False)

    st.subheader(result["title"])

    if is_meeting:
        st.caption("🏷️ Format: **Meeting Minutes** (Action items & decisions extracted)")
        tab_options = ["Summary", "Transcript", "Action Items", "Decisions", "Ask Question"]
    else:
        st.caption("🏷️ Format: **Video / Tutorial / Talk** (Core concepts & takeaways extracted)")
        tab_options = ["Summary", "Transcript", "Key Concepts & Takeaways", "Ask Question"]

    # Segmented navigation control ensures active view is NEVER reset by Q&A interactions
    default_tab = st.session_state.get("active_view")
    if default_tab not in tab_options:
        default_tab = "Summary"

    active_view = st.segmented_control(
        "Navigation",
        options=tab_options,
        default=default_tab,
        key="active_view",
        label_visibility="collapsed",
    )

    if active_view == "Summary":
        st.markdown("### Summary")
        st.markdown(result["summary"])

    elif active_view == "Transcript":
        st.markdown("### Full Transcript")
        st.text_area("", result["transcript"], height=320, label_visibility="collapsed")

    elif active_view == "Action Items":
        st.markdown("### Action Items")
        st.markdown(result.get("action_items") or "No action items extracted.")

    elif active_view == "Decisions":
        st.markdown("### Key Decisions")
        st.markdown(result.get("decisions") or "No key decisions recorded.")

    elif active_view == "Key Concepts & Takeaways":
        st.markdown("### Key Concepts & Takeaways")
        st.markdown(result.get("takeaways") or "No takeaways extracted.")

    elif active_view == "Ask Question":
        st.markdown("### 💬 Ask Questions from the Content (RAG)")
        st.caption("Ask anything about the video/transcript. Each response is grounded in the retrieved transcript chunks.")

        if "qa_history" not in st.session_state:
            st.session_state["qa_history"] = []

        # Display conversational chat history
        for chat in st.session_state["qa_history"]:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        # Input form that automatically clears itself on submit so the asking bar is immediately ready again!
        with st.form("ask_question_form", clear_on_submit=True):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_query = st.text_input(
                    "Ask a question:",
                    placeholder="Type your question here and press Enter (or click Ask)...",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

        if submitted and user_query.strip():
            with st.chat_message("user"):
                st.write(user_query)

            # Check if this is a polite conversational greeting or acknowledgment
            courtesy = handle_conversational_query(user_query)
            if courtesy:
                ans = courtesy
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Searching transcript and generating answer..."):
                        docs = query_transcript(user_query, k=4)
                        if not docs:
                            ans = "That is not mentioned in this video."
                        else:
                            chain = load_rag_chain()
                            ans = chain.invoke({"documents": docs, "question": user_query})

            with st.chat_message("assistant"):
                st.write(ans)

            st.session_state["qa_history"].append({"question": user_query, "answer": ans})
            st.rerun()

        if st.session_state.get("qa_history"):
            st.markdown("---")
            if st.button("🗑️ Clear Q&A History"):
                st.session_state["qa_history"] = []
                st.rerun()

else:
    st.info(
        "👈 **Get started from the sidebar on the left:**\n\n"
        "1. Select an input mode: **Upload audio/video**, **YouTube URL**, or **Paste transcript**.\n"
        "2. *(Tip)* You can click **'Sample Tutorial'** or **'Sample Meeting'** to test immediately!\n"
        "3. Click **Process transcript**.\n"
        "4. The intelligent summary and conversational Q&A will appear right here."
    )
