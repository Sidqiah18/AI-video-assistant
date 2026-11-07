import os
import re
import tempfile
from typing import Optional

import requests
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_cookie_file() -> Optional[str]:
    """Retrieve cookies file path from disk or from Streamlit/environment secrets."""
    # 1. Local cookies.txt in workspace
    local_cookies = os.path.join(os.getcwd(), "cookies.txt")
    if os.path.isfile(local_cookies):
        return local_cookies

    # 2. Check Streamlit secrets or environment variable
    raw_cookies = None
    try:
        import streamlit as st
        raw_cookies = st.secrets.get("YOUTUBE_COOKIES")
    except Exception:
        pass

    if not raw_cookies:
        raw_cookies = os.getenv("YOUTUBE_COOKIES")

    if raw_cookies and raw_cookies.strip():
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8")
        tmp.write(raw_cookies)
        tmp.close()
        return tmp.name

    return None


def fetch_youtube_transcript(url: str) -> Optional[str]:
    """
    Directly extracts official or auto-generated captions from YouTube without downloading media streams.
    Bypasses datacenter/cloud HTTP 403 Forbidden errors and executes in 1-2 seconds.
    """
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    cookie_path = get_cookie_file()
    if cookie_path:
        ydl_opts["cookiefile"] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None

            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}

            # Prioritize English caption tracks, or fallback to first available language track
            caption_tracks = (
                subs.get("en")
                or subs.get("en-US")
                or subs.get("en-GB")
                or auto_subs.get("en")
                or auto_subs.get("en-US")
                or auto_subs.get("en-GB")
            )

            if not caption_tracks:
                for lang, tracks in {**subs, **auto_subs}.items():
                    if tracks:
                        caption_tracks = tracks
                        break

            if not caption_tracks:
                return None

            # Look for json3 or vtt formats
            target_format = None
            for fmt in caption_tracks:
                if fmt.get("ext") == "json3":
                    target_format = fmt
                    break
            if not target_format and caption_tracks:
                target_format = caption_tracks[0]

            if not target_format or not target_format.get("url"):
                return None

            resp = requests.get(target_format["url"], timeout=10)
            if resp.status_code != 200:
                return None

            if target_format.get("ext") == "json3":
                data = resp.json()
                lines = []
                for event in data.get("events", []):
                    for seg in event.get("segs", []):
                        txt = seg.get("utf8", "")
                        if txt and txt != "\n":
                            lines.append(txt)
                raw_text = "".join(lines)
            else:
                raw_text = re.sub(r"<[^>]+>", "", resp.text)
                raw_text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> .*", "", raw_text)

            # Clean audio annotations like [Music] or [Applause]
            cleaned = re.sub(r"\[(Music|Applause|Laughter|Loud Music|Silence)\]", "", raw_text, flags=re.IGNORECASE)
            cleaned = " ".join(cleaned.split())
            return cleaned if cleaned.strip() else None

    except Exception:
        return None


def download_audio(url: str) -> str:
    """
    Downloads audio from a given URL and returns the actual downloaded file path.
    Uses mobile player client spoofing and automatic browser cookie fallback
    to prevent YouTube 'Sign in to confirm you’re not a bot' blocks.
    """
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios"],
            }
        },
    }

    cookie_file = get_cookie_file()
    if cookie_file:
        base_opts["cookiefile"] = cookie_file

    env_browser = os.getenv("YOUTUBE_COOKIES_BROWSER")
    if env_browser:
        base_opts["cookiesfrombrowser"] = (env_browser.strip().lower(),)

    expected_file = None
    try:
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise ValueError(f"No metadata returned for URL: {url}")
            expected_file = ydl.prepare_filename(info)
    except Exception as initial_exc:
        err_msg = str(initial_exc).lower()
        if "sign in to confirm" in err_msg or "bot" in err_msg or "403" in err_msg:
            fallback_success = False
            for browser in ("chrome", "edge", "firefox", "brave"):
                try:
                    fallback_opts = dict(base_opts)
                    fallback_opts["cookiesfrombrowser"] = (browser,)
                    with yt_dlp.YoutubeDL(fallback_opts) as fallback_ydl:
                        info = fallback_ydl.extract_info(url, download=True)
                        if info:
                            expected_file = fallback_ydl.prepare_filename(info)
                            fallback_success = True
                            break
                except Exception:
                    continue
            if not fallback_success:
                raise initial_exc
        else:
            raise initial_exc

    if expected_file and os.path.isfile(expected_file):
        return expected_file

    audio_files = []
    for file_name in os.listdir(DOWNLOAD_DIR):
        full_path = os.path.join(DOWNLOAD_DIR, file_name)
        if not os.path.isfile(full_path):
            continue
        if file_name.lower().endswith((".mp3", ".wav", ".m4a", ".webm", ".ogg", ".aac", ".flac", ".mp4")):
            audio_files.append(full_path)

    if not audio_files:
        raise FileNotFoundError(
            f"Downloaded audio file for '{url}' was not found in {DOWNLOAD_DIR}. "
            "Check that the YouTube URL is valid and accessible."
        )

    return max(audio_files, key=os.path.getmtime)


def convert_to_wav(input_file: str) -> str:
    """Converts an audio file to WAV format."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
    audio.export(output_file, format="wav")
    return output_file


def chunk_audio(input_file: str, chunk_length_minutes: int = 1) -> list:
    """Splits an audio file into smaller chunks."""
    audio = AudioSegment.from_wav(input_file)
    chunk_length_ms = chunk_length_minutes * 60 * 1000
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"{os.path.splitext(input_file)[0]}_chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    return chunks


def process_audio(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        downloaded_file = download_audio(source)
        wav_file = convert_to_wav(downloaded_file)
    else:
        wav_file = convert_to_wav(source)

    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"Processed audio file not found: {wav_file}")

    chunks = chunk_audio(wav_file)
    return chunks
