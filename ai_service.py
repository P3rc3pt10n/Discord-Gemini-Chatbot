"""
AI service layer for interacting with Google's Generative AI API.
Handles chat session management and response generation.
"""
import traceback
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from settings import (
    GOOGLE_AI_KEY,
    TEXT_GENERATION_CONFIG,
    SAFETY_SETTINGS,
    BOT_TEMPLATE
)
from storage import log_error


class AIService:
    """Manages interactions with Google's Generative AI API."""
    
    def __init__(self):
        """Initialize the AI service with configuration."""
        self.client = genai.Client(api_key=GOOGLE_AI_KEY)
        self.model_name = "gemini-3-flash-preview"
        self.config = types.GenerateContentConfig(
            **TEXT_GENERATION_CONFIG,
            safety_settings=[types.SafetySetting(**s) for s in SAFETY_SETTINGS] or None
        )
        self.message_history: Dict[int, Any] = {}
    
    def _create_chat(self, history: List[Any]) -> Any:
        """Create a new async chat session seeded with the given history."""
        return self.client.aio.chats.create(
            model=self.model_name,
            config=self.config,
            history=history
        )
    
    def load_history(self, history_data: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Load previously saved chat history.
        
        Args:
            history_data: Dictionary mapping channel IDs to chat histories
        """
        for channel_id, history in history_data.items():
            self.message_history[channel_id] = self._create_chat(history)
    
    async def generate_response(self, channel_id: int, attachments: List[Dict[str, Any]], text: str) -> str:
        """
        Generate a response from the AI model for the given input.
        
        Args:
            channel_id: Discord channel ID for context
            attachments: List of attachment data dictionaries
            text: The user's message text
            
        Returns:
            The AI model's response text
            
        Raises:
            Exception: If the API call fails
        """
        response: Optional[Any] = None
        try:
            # Prepare prompt parts
            # Attachments are {"mime_type", "data"} dicts; the new SDK needs Part objects
            prompt_parts: List[Any] = [
                types.Part.from_bytes(data=a["data"], mime_type=a["mime_type"])
                for a in attachments
            ]
            prompt_parts.append(text)
            
            # Initialize chat session if not exists
            if channel_id not in self.message_history:
                self.message_history[channel_id] = self._create_chat(BOT_TEMPLATE)
            
            # Send message to AI
            response = await self.message_history[channel_id].send_message(prompt_parts)
            return response.text if response else ""
            
        except Exception as e:
            # Log detailed error information for debugging
            try:
                history_info = str(self.message_history[channel_id].get_history())
                candidates = str(response.candidates) if response else "N/A"
                parts = (
                    str(response.candidates[0].content.parts)
                    if response and response.candidates and response.candidates[0].content
                    else "N/A"
                )
                prompt_feedbacks = str(response.prompt_feedback) if response else "N/A"
            except:
                history_info = "N/A"
                candidates = "N/A"
                parts = "N/A"
                prompt_feedbacks = "N/A"
            
            log_error(
                text=text,
                error_traceback=traceback.format_exc(),
                history=history_info,
                candidates=candidates,
                parts=parts,
                prompt_feedbacks=prompt_feedbacks
            )
            raise
    
    def reset_channel_history(self, channel_id: int, custom_template: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Reset the chat history for a channel.
        
        Args:
            channel_id: Discord channel ID
            custom_template: Optional custom initial template for the chat
        """
        if custom_template is None:
            custom_template = BOT_TEMPLATE
        
        self.message_history[channel_id] = self._create_chat(custom_template)
    
    def delete_channel_history(self, channel_id: int) -> None:
        """
        Delete chat history for a channel.
        
        Args:
            channel_id: Discord channel ID
        """
        if channel_id in self.message_history:
            del self.message_history[channel_id]
    
    def get_history(self, channel_id: int) -> List[Dict[str, Any]]:
        """
        Get the chat history for a channel.
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            The chat history list
        """
        if channel_id in self.message_history:
            return self.message_history[channel_id].get_history()
        return []
