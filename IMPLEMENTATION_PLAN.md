# Implementation Plan: Adding CUDA Support via llama.cpp/vLLM

## Quick Answer

**Yes, it's technically possible** to add llama.cpp or vLLM support, but it requires **significant architectural changes** to exo. Here's what's involved:

## What Needs to Change

### 1. **Engine Abstraction Layer** (New)
- Create abstract interfaces for Model, Tokenizer, Sampler, Engine
- Refactor MLX engine to use these interfaces
- Create new llama.cpp/vLLM engines implementing the same interfaces

### 2. **Model Format Handling** (Major Change)
- **Current**: MLX safetensors format
- **llama.cpp needs**: GGUF format
- **vLLM needs**: Standard PyTorch/HuggingFace format
- Solution: Download appropriate format based on engine, or convert formats

### 3. **Runner Refactoring** (Medium Change)
- Currently: `runner.py` directly calls `initialize_mlx()`, `mlx_generate()`
- Need: Engine selection logic, route to appropriate engine
- Example: `engine = get_engine(config.engine_type)` then `engine.initialize()`

### 4. **Distributed Parallelism** (Complex)
- **MLX**: Built-in distributed groups (`mx.distributed.Group`)
- **llama.cpp**: CUDA multi-GPU (simpler, single process)
- **vLLM**: Tensor parallelism built-in, but different model
- Challenge: Each handles distribution differently

### 5. **Configuration** (Simple)
- Add engine type to model metadata/config
- Add engine-specific config options (GPU layers, quantization, etc.)

## Implementation Difficulty

- **Difficulty**: High (significant refactoring)
- **Time Estimate**: 2-3 months for full implementation
- **Risk**: Medium-High (touches core inference code)

## Recommendation

Given the complexity, I'd suggest:

1. **Short-term**: Use exo as-is with CPU mode, or wait for official MLX CUDA support
2. **Medium-term**: Create a proof-of-concept llama.cpp engine to validate approach
3. **Long-term**: Consider contributing to upstream exo project for official multi-engine support

## Alternative Approach: Simpler Integration

Instead of full abstraction, could:
- Keep MLX as-is
- Add llama.cpp/vLLM as separate, parallel implementation
- Route requests based on model config
- Less clean architecture but faster to implement (~1 month)

## Files Created

I've created proof-of-concept files showing the structure:
- `ENGINE_ABSTRACTION_PROPOSAL.md` - Detailed proposal
- `src/exo/worker/engines/base.py` - Abstract interfaces
- `src/exo/worker/engines/llamacpp/__init__.py` - Example llama.cpp skeleton

These demonstrate the concepts but are **not functional** - they show what the architecture would look like.

## Next Steps (If Proceeding)

1. **Evaluate feasibility**: Test llama.cpp with a simple model
2. **Prototype**: Implement minimal llama.cpp engine for single GPU
3. **Refactor**: Gradually abstract MLX to use base interfaces
4. **Integrate**: Wire up engine selection in runner
5. **Test**: Thorough testing with various models

Would you like me to start implementing a proof-of-concept, or would you prefer to wait for official CUDA support?

