# Proof of concept: Engine abstraction interface
# This demonstrates what would be needed to support multiple backends

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Protocol

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.tasks import ChatCompletionTaskParams


class Model(Protocol):
    """Abstract model interface that all engines must implement."""
    pass


class Tokenizer(Protocol):
    """Abstract tokenizer interface that all engines must implement."""
    bos_token: str | None
    eos_token_ids: list[int]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    
    def apply_chat_template(
        self,
        messages_dicts: list[dict[str, object]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str: ...


class Sampler(Protocol):
    """Abstract sampler interface for token generation."""
    def __call__(self, logits: object) -> int: ...


class Engine(ABC):
    """Abstract base class for inference engines (MLX, llama.cpp, vLLM, etc.)"""
    
    @abstractmethod
    def initialize(
        self, 
        bound_instance: BoundInstance,
    ) -> tuple[Model, Tokenizer, Sampler]:
        """
        Initialize the model, tokenizer, and sampler.
        
        Returns:
            Tuple of (model, tokenizer, sampler) objects
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate(
        self,
        model: Model,
        tokenizer: Tokenizer,
        sampler: Sampler,
        task: ChatCompletionTaskParams,
    ) -> Generator[GenerationResponse, None, None]:
        """
        Generate text based on the task parameters.
        
        Yields:
            GenerationResponse objects for streaming responses
        """
        raise NotImplementedError
    
    @abstractmethod
    def warmup(
        self,
        model: Model,
        tokenizer: Tokenizer,
        sampler: Sampler,
    ) -> int:
        """
        Warmup the model by generating some tokens.
        
        Returns:
            Number of tokens generated during warmup
        """
        raise NotImplementedError


# Engine registry - would be used to select engine based on config
_ENGINES: dict[str, type[Engine]] = {}


def register_engine(name: str, engine_class: type[Engine]) -> None:
    """Register an engine implementation."""
    _ENGINES[name] = engine_class


def get_engine(name: str) -> type[Engine]:
    """Get an engine class by name."""
    if name not in _ENGINES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(_ENGINES.keys())}")
    return _ENGINES[name]

