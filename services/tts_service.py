"""Text-to-Speech service using OpenAI TTS."""
import logging
from typing import Optional
import tempfile
from pathlib import Path
from openai import AsyncOpenAI
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service using OpenAI TTS API."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.voice = settings.openai_tts_voice
        self.model = settings.openai_tts_model
        logger.info(f"TTS Service initialized with voice: {self.voice}, model: {self.model}")
    
    async def synthesize(
        self, 
        text: str,
        output_path: Optional[str] = None,
    ) -> str:
        try:
            logger.info(f"Synthesizing speech: {text[:50]}...")
            
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3"
            )
            
            response.stream_to_file(output_path)
            
            logger.info(f"TTS synthesis complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise
    
    async def synthesize_stream(
        self,
        text: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ):
        """
        Synthesize text to speech with streaming.
        
        Args:
            text: Text to synthesize
            stability: Voice stability
            similarity_boost: Voice similarity boost
        
        Yields:
            Audio chunks as they're generated
        """
        try:
            audio_generator = await self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                optimize_streaming_latency=4,  # Optimize for streaming
                output_format="mp3_44100_128",
                text=text,
                model_id=self.model_id,
                voice_settings=VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=0.0,
                    use_speaker_boost=True
                )
            )
            
            async for chunk in audio_generator:
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise
