import yt_dlp
from pydub import AudioSegment  
import os
DOWNLOAD_DIR="downloads/"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_audio(url: str) -> str:
    """
    Downloads audio from a given URL and saves it as an MP3 file.

    Args:
        url (str): The URL of the audio to download.

    Returns:
        str: The path to the downloaded MP3 file.
    """
    output_path="os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')

    return audio_file

def convert_to_wav(input_file: str) -> str:
    """
    Converts an audio file to WAV format.

    Args:
        input_file (str): The path to the input audio file.

    Returns:
        str: The path to the converted WAV file.
    """
    output_file = os.path.splitext(input_file)[0] + '_converted.wav'
    audio = AudioSegment.from_wav(input_file)
    audio=audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
    audio.export(output_file, format='wav')
    return output_file

def chunk_audio(input_file: str, chunk_length_ms: int = 10) -> list:
    """
    Splits an audio file into smaller chunks.

    Args:
        input_file (str): The path to the input audio file.
        chunk_length_ms (int): The length of each chunk in milliseconds. Default is 60000 ms (1 minute).

    Returns:
        list: A list of paths to the chunked audio files.
    """
    audio = AudioSegment.from_wav(input_file)
    chunk_length_ms = chunk_length_ms * 60 * 1000  # Convert seconds to milliseconds
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"{os.path.splitext(input_file)[0]}_chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_filename, format='wav')
        chunks.append(chunk_filename)
    return chunks

def process_audio(source: str) -> list:
    if source.startswith("http") or source.startswith("https"):
        wav_file = download_audio(source)
    else:
        wav_file = convert_to_wav(source)
    print(f"Audio file processed: {wav_file}")
    chunks = chunk_audio(wav_file)
    print(f"Audio file chunked into {len(chunks)} parts.")
    return chunks
