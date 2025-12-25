# llama.cpp engine implementation for CUDA support

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from llama_cpp import Llama  # type: ignore

from exo.shared.types.api import ChatCompletionMessageText, FinishReason
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.download.download_utils import build_model_path
from exo.worker.runner.bootstrap import logger

try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

LLAMA_CPP_AVAILABLE = _LLAMA_CPP_AVAILABLE


class LlamaCppModel:
    """Wrapper for llama.cpp Llama model to match expected interface."""
    def __init__(self, llama: Any):  # type: ignore
        self.llama = llama


class LlamaCppTokenizer:
    """Wrapper for llama.cpp tokenizer to match expected interface."""
    def __init__(self, llama: Any):  # type: ignore
        self.llama = llama
        self.bos_token: str | None = llama.token_bos() if hasattr(llama, 'token_bos') else None  # type: ignore
        # Most models use token_eos() as EOS token
        self.eos_token_ids: list[int] = [llama.token_eos()] if hasattr(llama, 'token_eos') else []  # type: ignore
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        return self.llama.tokenize(text.encode("utf-8"), add_bos=add_special_tokens)  # type: ignore
    
    def apply_chat_template(
        self,
        messages_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply chat template. llama.cpp handles this internally,
        but we format messages here for compatibility.
        """
        formatted_messages = []
        for msg in messages_dicts:
            role = msg.get("role", "user")
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, list):
                # Handle multimodal content
                content = content[0].get("text", "") if content else ""
            formatted_messages.append({"role": role, "content": str(content)})
        
        # llama.cpp expects messages in a specific format
        # We'll format as simple prompt for now, can be improved
        prompt_parts = []
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        
        if add_generation_prompt:
            prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)


def _find_gguf_file(model_dir: Path) -> Path | None:
    """Find GGUF file in model directory."""
    if not model_dir.exists():
        return None
    
    # Look for .gguf files
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        return None
    
    # Prefer quantized models (Q4, Q5, Q8) over full precision
    priority_order = ["Q8", "Q6", "Q5", "Q4", "F16", "F32"]
    for priority in priority_order:
        for f in gguf_files:
            if priority in f.name:
                return f
    
    # Return first GGUF file found
    return gguf_files[0]


def initialize_llamacpp(
    bound_instance: BoundInstance,
    n_gpu_layers: int = -1,  # -1 means use all GPU layers
    n_ctx: int = 4096,
    verbose: bool = False,
) -> tuple[LlamaCppModel, LlamaCppTokenizer, Callable[[Any], int]]:
    """
    Initialize llama.cpp model, tokenizer, and sampler.
    
    Args:
        bound_instance: Bound instance with model metadata
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        n_ctx: Context window size
        verbose: Enable verbose logging
    
    Returns:
        Tuple of (model, tokenizer, sampler)
    """
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is not installed. "
            "Install with: pip install 'llama-cpp-python[cuda]' for CUDA support"
        )
    
    model_id = bound_instance.bound_shard.model_meta.model_id
    model_dir = build_model_path(model_id)
    
    logger.info(f"Looking for GGUF model in: {model_dir}")
    
    # Find GGUF file
    gguf_path = _find_gguf_file(model_dir)
    if gguf_path is None:
        raise FileNotFoundError(
            f"No GGUF file found in {model_dir}. "
            f"llama.cpp requires GGUF format models. "
            f"Download GGUF models from HuggingFace (e.g., TheBloke/*-GGUF repos)"
        )
    
    logger.info(f"Loading GGUF model from: {gguf_path}")
    
    # Initialize llama.cpp with CUDA support
    llama = Llama(  # type: ignore
        model_path=str(gguf_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
        # Use CUDA if available
        n_threads=None,  # Auto-detect
        use_mmap=True,
        use_mlock=False,
    )
    
    logger.info("llama.cpp model loaded successfully")
    
    model = LlamaCppModel(llama)
    tokenizer = LlamaCppTokenizer(llama)
    
    # Simple sampler function - llama.cpp handles sampling internally
    def sampler(logits: Any) -> int:
        # This is a placeholder - llama.cpp handles sampling in its generate method
        # We don't actually use this sampler
        return 0
    
    return model, tokenizer, sampler


def llamacpp_generate(
    model: LlamaCppModel,
    tokenizer: LlamaCppTokenizer,
    sampler: Callable[[Any], int],  # Not used, but kept for interface compatibility
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate text using llama.cpp.
    
    Yields:
        GenerationResponse objects for streaming responses
    """
    llama = model.llama
    
    # Format messages for chat completion
    messages = []
    for msg in task.messages:
        if isinstance(msg.content, ChatCompletionMessageText):
            content = msg.content.text
        elif isinstance(msg.content, list):
            content = msg.content[0].text if msg.content else ""
        else:
            content = str(msg.content) if msg.content else ""
        
        messages.append({
            "role": msg.role,
            "content": content
        })
    
    # llama.cpp's create_chat_completion handles the chat formatting
    max_tokens = task.max_tokens or 512
    temperature = task.temperature if task.temperature is not None else 0.7
    top_p = task.top_p if task.top_p is not None else 0.9
    
    logger.info(f"Generating with llama.cpp: max_tokens={max_tokens}, temperature={temperature}")
    
    # Use create_chat_completion for proper chat formatting
    stream = llama.create_chat_completion(  # type: ignore
        messages=messages,  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        stop=task.stop if isinstance(task.stop, list) else [task.stop] if task.stop else None,
    )
    
    token_count = 0
    for chunk in stream:
        if "choices" not in chunk or not chunk["choices"]:
            continue
        
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        content = delta.get("content", "")
        
        # Map llama.cpp finish_reason to our FinishReason (which is a Literal type)
        mapped_finish_reason: FinishReason | None = None
        if finish_reason == "stop":
            mapped_finish_reason = "stop"
        elif finish_reason == "length":
            mapped_finish_reason = "length"
        elif finish_reason == "tool_calls":
            mapped_finish_reason = "tool_calls"
        
        if content:
            token_count += 1
            yield GenerationResponse(
                text=content,
                token=token_count,
                finish_reason=mapped_finish_reason,
            )
        
        if finish_reason:
            break


def warmup_llamacpp(
    model: LlamaCppModel,
    tokenizer: LlamaCppTokenizer,
    sampler: Callable[[Any], int],
) -> int:
    """
    Warmup llama.cpp model by generating some tokens.
    
    Returns:
        Number of tokens generated during warmup
    """
    llama = model.llama
    
    warmup_prompt = "Prompt to warm up the inference engine. Repeat this."
    
    logger.info("Warming up llama.cpp model")
    
    tokens_generated = 0
    stream = llama(  # type: ignore
        warmup_prompt,
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )
    
    for chunk in stream:  # type: ignore
        if "choices" in chunk and chunk["choices"]:  # type: ignore
            delta = chunk["choices"][0].get("delta", {})  # type: ignore
            if delta.get("content"):  # type: ignore
                tokens_generated += 1
    
    logger.info(f"Warmed up llama.cpp model by generating {tokens_generated} tokens")
    
    return tokens_generated

