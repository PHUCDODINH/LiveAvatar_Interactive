"""Speech-to-Text service using OpenAI Whisper."""
import asyncio
import logging
from pathlib import Path
from typing import Optional
import tempfile

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-Text service using OpenAI Whisper API."""
    
    def __init__(self):
        """Initialize STT service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_whisper_model
        logger.info(f"STT Service initialized with model: {self.model}")
    
    async def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio bytes in supported format (mp3, wav, etc.)
            language: Optional language code (e.g., 'en', 'zh')
        
        Returns:
            Transcribed text
        """
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # Transcribe using OpenAI Whisper
            with open(temp_audio_path, 'rb') as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            
            # Clean up temp file
            Path(temp_audio_path).unlink()
            
            logger.info(f"Transcription successful: {transcript[:50]}...")
            return transcript
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise
    
    async def transcribe_file(self, file_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            file_path: Path to audio file
            language: Optional language code
        
        Returns:
            Transcribed text
        """
        try:
            with open(file_path, 'rb') as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            
            logger.info(f"File transcription successful: {transcript[:50]}...")
            return transcript
            
        except Exception as e:
            logger.error(f"STT file transcription error: {e}")
            raise
