import argparse
import os
import sys

# Fix Windows console charmap encoding issues with unicode in prints/logs
for stream in (sys.stdout, sys.stderr):
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from dotenv import load_dotenv

from utils.audio_processor import process_audio
from utils.core.rag_engine import load_rag_chain
from utils.core.transcriber import transcribe_all
from utils.core.vector_store import add_transcript, query_transcript
from utils.extractor import (
    extract_action_items,
    extract_decisions,
    extract_key_decisions,
    extract_key_takeaways,
    is_meeting_transcript,
)
from utils.summarizer import generate_title, summarize_transcript

load_dotenv()


def run_transcript_pipeline(
    source: str,
    translate: bool = False,
    index_in_vector_db: bool = True,
    force_meeting: bool = False,
):
    """Process audio/video input, transcribe it, summarize it, and index it for RAG."""
    print(f"Processing source: {source}")

    audio_chunks = process_audio(source)
    transcript = transcribe_all(audio_chunks, translate=translate)

    is_meeting = force_meeting or is_meeting_transcript(transcript)
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

    if index_in_vector_db:
        add_transcript(
            transcript,
            metadata={"title": title, "source": source, "is_meeting": str(is_meeting)},
        )

    return {
        "source": source,
        "title": title,
        "is_meeting": is_meeting,
        "transcript": transcript,
        "summary": summary,
        "takeaways": takeaways,
        "action_items": action_items,
        "decisions": decisions,
    }


def ask_question(question: str, k: int = 4):
    """Retrieve transcript chunks and answer a question using the RAG chain."""
    docs = query_transcript(question, k=k)
    if not docs:
        return "No transcript data is indexed yet. Run the transcription pipeline first."

    chain = load_rag_chain()
    return chain.invoke({"documents": docs, "question": question})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI video assistant transcript + RAG pipeline")
    parser.add_argument("source", help="Audio/video file path or YouTube URL")
    parser.add_argument("--question", help="Ask a question after transcript indexing")
    parser.add_argument("--translate", action="store_true", help="Translate the transcript to English")
    parser.add_argument("--meeting", action="store_true", help="Force meeting mode (extract action items & decisions)")
    parser.add_argument("--skip-index", action="store_true", help="Do not store transcript chunks in the vector DB")

    args = parser.parse_args()

    result = run_transcript_pipeline(
        args.source,
        translate=args.translate,
        index_in_vector_db=not args.skip_index,
        force_meeting=args.meeting,
    )

    print("\n=== Title ===")
    print(result["title"])
    print(f"Format: {'Meeting Minutes' if result['is_meeting'] else 'Video / Tutorial'}")
    print("\n=== Summary ===")
    print(result["summary"])

    if result["is_meeting"]:
        print("\n=== Action Items ===")
        print(result["action_items"])
        print("\n=== Key Decisions ===")
        print(result["decisions"])
    else:
        print("\n=== Key Concepts & Takeaways ===")
        print(result["takeaways"])

    if args.question:
        print("\n=== Answer ===")
        print(ask_question(args.question))
