"""Language Model service using OpenAI."""
import asyncio
import logging
from typing import List, Dict, Optional

from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Language Model service using OpenAI GPT."""
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize LLM service.
        
        Args:
            system_prompt: Optional system prompt to set avatar personality
        """
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        
        # Default system prompt for avatar
        self.system_prompt = system_prompt or (
            "You are a friendly and helpful AI assistant appearing as a virtual avatar. "
            "Keep your responses concise (2-3 sentences max) and conversational. "
            "Be warm, engaging, and natural in your interactions."
        )
        
        logger.info(f"LLM Service initialized with model: {self.model}")
    
    async def generate_response(
        self, 
        user_message: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response to user message.
        
        Args:
            user_message: The user's input text
            conversation_history: Optional list of previous messages
        
        Returns:
            AI assistant's response
        """
        try:
            # Build messages list
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7,
                stream=False
            )
            
            assistant_message = response.choices[0].message.content
            logger.info(f"LLM response generated: {assistant_message[:50]}...")
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    async def generate_response_stream(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Generate streaming response to user message.
        
        Args:
            user_message: The user's input text
            conversation_history: Optional list of previous messages
        
        Yields:
            Chunks of the response as they're generated
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if conversation_history:
                messages.extend(conversation_history)
            
            messages.append({"role": "user", "content": user_message})
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.7,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt for avatar personality."""
        self.system_prompt = new_prompt
        logger.info("System prompt updated")
