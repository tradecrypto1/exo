# Engine Abstraction Proposal for CUDA Support

## Current State

exo is tightly coupled to MLX:
- `runner.py` directly imports MLX functions (`initialize_mlx`, `mlx_generate`)
- Uses MLX-specific types (`mx.array`, `mx.distributed.Group`)
- Model format: MLX safetensors
- Distributed parallelism: MLX's built-in distributed groups

## Proposed Solution

Add an engine abstraction layer to support multiple backends (MLX, llama.cpp, vLLM).

## Architecture Changes Needed

### 1. Engine Interface/Protocol

Create abstract base classes for:
- `Model` - model interface
- `Tokenizer` - tokenizer interface  
- `Engine` - main engine interface (initialize, generate, warmup)

### 2. Engine Implementations

```
src/exo/worker/engines/
├── __init__.py
├── base.py          # Abstract interfaces
├── mlx/            # Existing MLX engine (refactor to use base)
├── llamacpp/       # New llama.cpp engine
└── vllm/           # New vLLM engine
```

### 3. Engine Selection

- Add engine type to model metadata/config
- Route to appropriate engine based on config
- Handle model format conversion if needed

## Challenges

1. **Model Format Differences**:
   - MLX: safetensors (MLX format)
   - llama.cpp: GGUF format
   - vLLM: Standard PyTorch/HuggingFace

2. **API Differences**:
   - Each backend has different initialization, generation APIs
   - Different tensor types and memory management

3. **Distributed Parallelism**:
   - MLX: Built-in distributed groups
   - llama.cpp: Single process, multi-GPU via CUDA
   - vLLM: Tensor parallelism built-in, different model

4. **Performance Characteristics**:
   - Different memory usage patterns
   - Different warmup requirements
   - Different quantization support

## Implementation Approach

### Phase 1: Proof of Concept (llama.cpp)
1. Create engine abstraction interfaces
2. Implement llama.cpp engine wrapper
3. Add model format conversion (safetensors → GGUF) or download GGUF directly
4. Modify runner to support engine selection
5. Test with single GPU

### Phase 2: Production Ready
1. Add vLLM engine
2. Handle distributed parallelism for each engine
3. Optimize model format handling
4. Add engine-specific configuration options

## Estimated Effort

- **Phase 1 (llama.cpp PoC)**: 2-4 weeks
- **Phase 2 (Production)**: 4-8 weeks
- **Total**: Significant architectural change requiring thorough testing

## Alternative: Simpler Integration

Instead of full abstraction, could:
1. Add llama.cpp/vLLM as separate runner types
2. Keep MLX as-is
3. Route requests to appropriate runner based on model config
4. Less clean but faster to implement

## Recommendation

Given the scope, I recommend:
1. Start with a proof-of-concept for llama.cpp (simpler than vLLM)
2. Test feasibility with single GPU
3. Evaluate if full abstraction is worth it vs. simpler parallel implementation
4. Consider contributing upstream to exo project

