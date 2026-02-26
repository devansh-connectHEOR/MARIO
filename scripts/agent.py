from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from typing import Any, List, Optional

class image_agent(ChatOpenAI):
    """
    A wrapper that intercepts messages and moves images from 
    'tool' messages to 'user' messages for OpenAI compatibility.
    """
    def _convert_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        fixed_messages = []
        for msg in messages:
            # Check if this is a ToolMessage containing image data
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                # 1. Extract the text (OpenAI allows text in ToolMessages)
                text_blocks = [b for b in msg.content if b.get("type") == "text"]
                # 2. Extract the images
                image_blocks = [b for b in msg.content if b.get("type") == "image_url"]
                
                # Update original message to be text-only
                msg.content = text_blocks if text_blocks else "Image retrieved."
                fixed_messages.append(msg)
                
                # Add a NEW HumanMessage for the images
                if image_blocks:
                    fixed_messages.append(HumanMessage(content=image_blocks))
            else:
                fixed_messages.append(msg)
        return fixed_messages

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
        # Fix messages before calling the real OpenAI invoke
        if isinstance(input, list):
            input = self._convert_messages(input)
        return super().invoke(input, config, **kwargs)