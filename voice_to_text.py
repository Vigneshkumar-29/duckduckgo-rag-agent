# voice_to_text.py
"""
AssemblyAI Voice-to-Text Module for DuckDuckGo AI RAG Agent.

Provides voice and audio transcription capabilities using the AssemblyAI SDK.
Supports audio files, microphone recordings (bytes / buffers), and remote URLs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

from dotenv import load_dotenv
import assemblyai as aai

# Load environment variables from .env file if present
load_dotenv()


# ==========================================
# API KEY MANAGEMENT
# ==========================================

def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """
    Retrieve the AssemblyAI API key from parameters, settings, or environment variables.

    Priority:
    1. Explicitly passed `api_key` argument
    2. aai.settings.api_key (if already set in-memory)
    3. Environment variable `ASSEMBLYAI_API_KEY`
    4. Streamlit secrets (if running in a Streamlit context)

    Returns:
        The API key string if found, otherwise None.
    """
    if api_key and api_key.strip():
        return api_key.strip()

    if aai.settings.api_key and aai.settings.api_key.strip():
        return aai.settings.api_key.strip()

    env_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # Attempt to load from Streamlit secrets if Streamlit is active
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ASSEMBLYAI_API_KEY" in st.secrets:
            key = st.secrets["ASSEMBLYAI_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    return None


def set_api_key(api_key: str) -> None:
    """
    Set the AssemblyAI API key for the current process and AssemblyAI settings.

    Args:
        api_key: The AssemblyAI API key string.
    """
    clean_key = api_key.strip()
    aai.settings.api_key = clean_key
    os.environ["ASSEMBLYAI_API_KEY"] = clean_key


def is_api_key_configured(api_key: Optional[str] = None) -> bool:
    """
    Check whether an AssemblyAI API key is available.

    Returns:
        True if an API key is available, False otherwise.
    """
    key = get_api_key(api_key)
    return bool(key and len(key) > 5)


def _ensure_api_key(api_key: Optional[str] = None) -> str:
    """
    Validate that an API key is set and configure aai.settings.api_key.
    Raises ValueError with actionable instructions if missing.
    """
    key = get_api_key(api_key)
    if not key:
        raise ValueError(
            "AssemblyAI API key not found.\n\n"
            "Please provide your API key using one of the following methods:\n"
            "1. Add ASSEMBLYAI_API_KEY=your_key_here to a .env file in the project root.\n"
            "2. Set the environment variable: export ASSEMBLYAI_API_KEY='your_key_here' "
            "(or set ASSEMBLYAI_API_KEY=... in Windows PowerShell/CMD).\n"
            "3. Pass api_key directly to transcribe_audio(..., api_key='your_key_here').\n"
            "4. Call set_api_key('your_key_here') before transcribing."
        )
    aai.settings.api_key = key
    return key


# ==========================================
# TRANSCRIPTION FUNCTIONS
# ==========================================

def transcribe_audio(
    audio_source: Union[str, Path, bytes, BinaryIO, Any],
    api_key: Optional[str] = None,
    config: Optional[aai.TranscriptionConfig] = None,
    language_code: Optional[str] = None,
    punctuate: bool = True,
    format_text: bool = True,
    poll_timeout: Optional[float] = None,
) -> str:
    """
    Transcribe an audio source into text using AssemblyAI.

    Args:
        audio_source: Local file path, URL, raw bytes, or file-like buffer (e.g. Streamlit UploadedFile).
        api_key: Optional AssemblyAI API key. If not provided, loaded from environment/.env.
        config: Optional custom `aai.TranscriptionConfig`. If provided, language_code, punctuate,
                and format_text parameters will be merged or overridden.
        language_code: Optional language code (e.g., 'en', 'es', 'fr'). Defaults to None (auto-detect or English).
        punctuate: Whether to include punctuation in transcription. Defaults to True.
        format_text: Whether to format numbers, dates, etc. Defaults to True.
        poll_timeout: Optional timeout in seconds to wait for transcription.

    Returns:
        The transcribed text string.

    Raises:
        ValueError: If the API key is missing.
        RuntimeError: If AssemblyAI transcription fails or returns an error.
    """
    _ensure_api_key(api_key)

    # Normalize audio source
    source_to_send = audio_source
    if hasattr(audio_source, "getvalue"):
        # Streamlit UploadedFile or BytesIO
        source_to_send = audio_source.getvalue()
    elif hasattr(audio_source, "read") and not isinstance(audio_source, (str, Path)):
        source_to_send = audio_source.read()
    elif isinstance(audio_source, Path):
        source_to_send = str(audio_source)

    # Build default config if not supplied
    if config is None:
        config_kwargs: Dict[str, Any] = {
            "punctuate": punctuate,
            "format_text": format_text,
        }
        if language_code:
            config_kwargs["language_code"] = language_code
        config = aai.TranscriptionConfig(**config_kwargs)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        data=source_to_send,
        config=config,
        poll_timeout=poll_timeout,
    )

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

    return transcript.text or ""


def transcribe_audio_detailed(
    audio_source: Union[str, Path, bytes, BinaryIO, Any],
    api_key: Optional[str] = None,
    config: Optional[aai.TranscriptionConfig] = None,
    speaker_labels: bool = False,
    language_code: Optional[str] = None,
    poll_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Transcribe an audio source and return detailed metadata including confidence,
    words, utterances, and duration.

    Args:
        audio_source: Local file path, URL, raw bytes, or file-like buffer.
        api_key: Optional AssemblyAI API key.
        config: Optional custom TranscriptionConfig.
        speaker_labels: Enable speaker diarization (who spoke when).
        language_code: Optional language code.
        poll_timeout: Optional poll timeout.

    Returns:
        Dictionary containing:
        - "text": The full transcribed text
        - "confidence": Overall confidence score
        - "audio_duration": Duration of the audio in seconds
        - "words": List of word objects with timestamps
        - "utterances": List of speaker-labeled turns (if speaker_labels=True)
        - "status": Transcript status
        - "transcript_id": The AssemblyAI transcript ID
    """
    _ensure_api_key(api_key)

    source_to_send = audio_source
    if hasattr(audio_source, "getvalue"):
        source_to_send = audio_source.getvalue()
    elif hasattr(audio_source, "read") and not isinstance(audio_source, (str, Path)):
        source_to_send = audio_source.read()
    elif isinstance(audio_source, Path):
        source_to_send = str(audio_source)

    if config is None:
        config_kwargs: Dict[str, Any] = {
            "speaker_labels": speaker_labels,
        }
        if language_code:
            config_kwargs["language_code"] = language_code
        config = aai.TranscriptionConfig(**config_kwargs)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        data=source_to_send,
        config=config,
        poll_timeout=poll_timeout,
    )

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

    utterances_data = []
    if transcript.utterances:
        for u in transcript.utterances:
            utterances_data.append({
                "speaker": u.speaker,
                "text": u.text,
                "start": u.start,
                "end": u.end,
                "confidence": u.confidence,
            })

    words_data = []
    if transcript.words:
        for w in transcript.words:
            words_data.append({
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
                "speaker": getattr(w, "speaker", None),
            })

    return {
        "text": transcript.text or "",
        "confidence": transcript.confidence,
        "audio_duration": transcript.audio_duration,
        "words": words_data,
        "utterances": utterances_data,
        "status": transcript.status.value,
        "transcript_id": transcript.id,
    }


def transcribe_file(file_path: Union[str, Path], api_key: Optional[str] = None) -> str:
    """
    Convenience function to transcribe an audio file from a local path.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    return transcribe_audio(audio_source=str(path_obj), api_key=api_key)


def transcribe_bytes(audio_bytes: bytes, api_key: Optional[str] = None) -> str:
    """
    Convenience function to transcribe audio directly from raw bytes.
    """
    return transcribe_audio(audio_source=audio_bytes, api_key=api_key)


# ==========================================
# OOP WRAPPER CLASS
# ==========================================

class VoiceTranscriber:
    """
    Object-oriented client for AssemblyAI speech-to-text operations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = get_api_key(api_key)
        if self.api_key:
            set_api_key(self.api_key)

    @property
    def is_configured(self) -> bool:
        """Check if an API key is currently set."""
        return is_api_key_configured(self.api_key)

    def set_key(self, api_key: str) -> None:
        """Update or set the API key."""
        self.api_key = api_key
        set_api_key(api_key)

    def transcribe(
        self,
        audio_source: Union[str, Path, bytes, BinaryIO, Any],
        **kwargs: Any,
    ) -> str:
        """Transcribe audio source into text."""
        return transcribe_audio(audio_source=audio_source, api_key=self.api_key, **kwargs)

    def transcribe_detailed(
        self,
        audio_source: Union[str, Path, bytes, BinaryIO, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transcribe audio source with full metadata."""
        return transcribe_audio_detailed(audio_source=audio_source, api_key=self.api_key, **kwargs)


# ==========================================
# CLI / STANDALONE TEST
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("AssemblyAI Voice-to-Text Module")
    print("=" * 60)

    configured = is_api_key_configured()
    if configured:
        key = get_api_key()
        masked_key = key[:4] + "..." + key[-4:] if key and len(key) > 8 else "***"
        print(f"Status: API Key is configured ({masked_key})")
    else:
        print("Status: API Key NOT configured.")
        print("Set ASSEMBLYAI_API_KEY in your .env file or environment.")

    if len(sys.argv) > 1:
        target_audio = sys.argv[1]
        print(f"\nAttempting transcription of: {target_audio}")
        try:
            result = transcribe_audio(target_audio)
            print("\nTranscription Result:")
            print("-" * 40)
            print(result)
            print("-" * 40)
        except Exception as exc:
            print(f"\nError: {exc}")
    else:
        print("\nUsage:")
        print("  python voice_to_text.py <path_or_url_to_audio>")
        print("\nSupported formats: WAV, MP3, M4A, OGG, FLAC, WEBM, AAC, etc.")
