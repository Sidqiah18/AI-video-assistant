import os
import re

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_audio(url: str) -> str:
    """
    Downloads audio from a given URL and returns the actual downloaded file path.
    Uses mobile player client spoofing and automatic browser cookie fallback
    to prevent YouTube 'Sign in to confirm you’re not a bot' blocks.

    Args:
        url (str): The URL of the audio to download.

    Returns:
        str: The path to the downloaded audio file.
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
                "player_client": ["ios", "android", "mweb"],
            }
        },
    }

    # If cookies.txt is provided in project root, use it
    cookies_file = os.path.join(os.getcwd(), "cookies.txt")
    if os.path.isfile(cookies_file):
        base_opts["cookiefile"] = cookies_file

    # If environment variable specifies a browser for cookies, use it
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
        # If blocked by YouTube bot detection, try automatic browser cookies fallback
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
    """
    Converts an audio file to WAV format.

    Args:
        input_file (str): The path to the input audio file.

    Returns:
        str: The path to the converted WAV file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    output_file = os.path.splitext(input_file)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
    audio.export(output_file, format="wav")
    return output_file


def chunk_audio(input_file: str, chunk_length_minutes: int = 1) -> list:
    """
    Splits an audio file into smaller chunks.

    Args:
        input_file (str): The path to the input audio file.
        chunk_length_minutes (int): Length of each chunk in minutes. Defaults to 1 minute.

    Returns:
        list: A list of paths to the chunked audio files.
    """
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
