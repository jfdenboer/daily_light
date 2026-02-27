"""TTS service exports."""

from .dialogue_synthesizer import DialogueSynthesizer, DialogueTurn
from .elevenlabs_tts_client import ElevenLabsTTSClient
from .speech_synthesizer import SpeechSynthesizer

__all__ = [
    "DialogueSynthesizer",
    "DialogueTurn",
    "ElevenLabsTTSClient",
    "SpeechSynthesizer",
]