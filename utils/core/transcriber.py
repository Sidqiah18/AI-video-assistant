import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL") or "base"  # You can change this to "tiny", "small", "medium", or "large" based on your needs
_model= None

def get_whisper_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model

def transcribe_audio(audio_file: str, translate: bool = False) -> str:
    """
    Transcribes an audio file using the Whisper model.

    Args:
        audio_file (str): The path to the audio file to be transcribed.
        translate (bool): Whether to translate the transcription to English.

    Returns:
        str: The transcribed text.
    """
    model = get_whisper_model()
    model.options['task'] = 'translate' if translate else 'transcribe'
    result = model.transcribe(audio_file)
    return result['text']