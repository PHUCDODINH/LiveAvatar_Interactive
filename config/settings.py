"""Configuration management for Interactive Avatar."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str
    elevenlabs_api_key: str
    
    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    # LiveAvatar
    liveavatar_ckpt_dir: str = "ckpt/Wan2.2-S2V-14B/"
    liveavatar_lora_path: str = "Quark-Vision/Live-Avatar"
    liveavatar_size: str = "704*384"  # Balanced quality for H100 80GB
    liveavatar_infer_frames: int = 32  # 0.67s videos (better lip sync)
    liveavatar_sample_steps: int = 2
    enable_fp8: bool = False
    enable_compile: bool = True
    default_avatar_prompt: str = "A person speaking naturally"
    default_avatar_image: str = "examples/man.png"
    
    # OpenAI
    openai_model: str = "gpt-3.5-turbo"
    openai_whisper_model: str = "whisper-1"
    openai_max_tokens: int = 150
    
    # OpenAI TTS
    openai_tts_model: str = Field(default="tts-1")
    openai_tts_voice: str = Field(default="alloy")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
