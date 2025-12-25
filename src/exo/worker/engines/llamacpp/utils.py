# Utility functions for llama.cpp engine

from pathlib import Path

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.worker.runner.bootstrap import logger


def apply_chat_template_llamacpp(
    messages: list,
    task: ChatCompletionTaskParams,
) -> str:
    """
    Apply chat template for llama.cpp.
    This formats messages into a prompt string.
    """
    formatted_parts = []
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        # Handle different content types
        if isinstance(content, ChatCompletionMessageText):
            content_text = content.text
        elif isinstance(content, list):
            content_text = content[0].text if content else ""
        else:
            content_text = str(content) if content else ""
        
        if not content_text:
            continue
        
        # Format based on role
        if role == "system":
            formatted_parts.append(f"System: {content_text}")
        elif role == "user":
            formatted_parts.append(f"User: {content_text}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content_text}")
        else:
            formatted_parts.append(f"{role.capitalize()}: {content_text}")
    
    # Add assistant prompt if needed
    formatted_parts.append("Assistant:")
    
    return "\n".join(formatted_parts)

