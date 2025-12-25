# Engine selector - determines which engine to use based on available model files

import os
from pathlib import Path
from typing import Any, Callable

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.runner.bootstrap import logger

# Try to import engines
MLX_AVAILABLE = False
mlx_generate = None
warmup_inference = None
initialize_mlx = None

try:
    # Try importing MLX core first to check if it's actually usable
    import mlx.core as mx  # type: ignore
    from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
    from exo.worker.engines.mlx.utils_mlx import initialize_mlx
    MLX_AVAILABLE = True
except (ImportError, OSError) as e:
    MLX_AVAILABLE = False
    logger.warning(f"MLX engine not available: {e}")

try:
    from exo.worker.engines.llamacpp import (
        initialize_llamacpp,
        llamacpp_generate,
        warmup_llamacpp,
    )
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    logger.warning("llama.cpp engine not available")


def _has_gguf_file(model_dir: Path) -> bool:
    """Check if model directory contains GGUF files."""
    if not model_dir.exists():
        return False
    return any(model_dir.glob("*.gguf"))


def select_engine(bound_instance: BoundInstance) -> str:
    """
    Select engine based on available model files.
    
    Priority:
    1. Use llama.cpp if GGUF files found
    2. Use MLX if available (default)
    3. Raise error if neither available
    
    Returns:
        Engine name: "llamacpp" or "mlx"
    """
    model_id = bound_instance.bound_shard.model_meta.model_id
    model_dir = build_model_path(model_id)
    
    # Check for GGUF files (llama.cpp format)
    if LLAMACPP_AVAILABLE and _has_gguf_file(model_dir):
        logger.info(f"Selected llama.cpp engine for {model_id} (GGUF files found)")
        return "llamacpp"
    
    # Default to MLX
    if MLX_AVAILABLE:
        logger.info(f"Selected MLX engine for {model_id}")
        return "mlx"
    
    # If MLX not available but model is MLX format, provide helpful error
    if not MLX_AVAILABLE and not _has_gguf_file(model_dir):
        raise RuntimeError(
            f"MLX engine is not available on this platform (Linux/Docker). "
            f"Model '{model_id}' requires MLX format (safetensors).\n"
            f"Options:\n"
            f"1. Use a GGUF model instead (download GGUF version to {model_dir}/)\n"
            f"2. Run on macOS where MLX is supported\n"
            f"3. Wait for MLX Linux CUDA support (currently in development)"
        )
    
    raise RuntimeError(
        "No available engines. "
        "Install MLX for MLX models or llama-cpp-python for GGUF models."
    )


def initialize_model(
    bound_instance: BoundInstance,
) -> tuple[Any, Any, Callable[[Any], Any]]:
    """
    Initialize model using the appropriate engine.
    
    Returns:
        Tuple of (model, tokenizer, sampler)
    """
    engine = select_engine(bound_instance)
    
    if engine == "llamacpp":
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama.cpp engine not available")
        return initialize_llamacpp(bound_instance)
    elif engine == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError("MLX engine not available")
        return initialize_mlx(bound_instance)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def generate_text(
    model: Any,
    tokenizer: Any,
    sampler: Callable[[Any], Any],
    task: Any,
    engine: str | None = None,
):
    """
    Generate text using the appropriate engine.
    
    Args:
        model: Model object (engine-specific)
        tokenizer: Tokenizer object (engine-specific)
        sampler: Sampler function (engine-specific)
        task: ChatCompletionTaskParams
        engine: Engine name (auto-detected if None)
    """
    if engine is None:
        # Try to detect from model type
        if hasattr(model, "llama"):  # llama.cpp model
            engine = "llamacpp"
        else:
            engine = "mlx"
    
    if engine == "llamacpp":
        return llamacpp_generate(model, tokenizer, sampler, task)
    elif engine == "mlx":
        return mlx_generate(model, tokenizer, sampler, task)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def warmup_model(
    model: Any,
    tokenizer: Any,
    sampler: Callable[[Any], Any],
    engine: str | None = None,
) -> int:
    """
    Warmup model using the appropriate engine.
    
    Args:
        model: Model object (engine-specific)
        tokenizer: Tokenizer object (engine-specific)
        sampler: Sampler function (engine-specific)
        engine: Engine name (auto-detected if None)
    
    Returns:
        Number of tokens generated during warmup
    """
    if engine is None:
        # Try to detect from model type
        if hasattr(model, "llama"):  # llama.cpp model
            engine = "llamacpp"
        else:
            engine = "mlx"
    
    if engine == "llamacpp":
        return warmup_llamacpp(model, tokenizer, sampler)
    elif engine == "mlx":
        return warmup_inference(model, tokenizer, sampler)
    else:
        raise ValueError(f"Unknown engine: {engine}")

