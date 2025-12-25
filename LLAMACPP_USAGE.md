# Using llama.cpp Engine for CUDA Support

This document explains how to use the llama.cpp engine proof-of-concept for GPU-accelerated inference on Linux/Windows.

## Overview

The llama.cpp engine provides CUDA support for exo when MLX's CPU-only mode isn't sufficient. It automatically detects GGUF model files and uses llama.cpp for inference.

## Installation

### 1. Install llama-cpp-python with CUDA support

For CUDA support on Linux/Windows:

```bash
# Option 1: Using pip (recommended)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Option 2: Using uv (if using uv for package management)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" uv pip install llama-cpp-python
```

**Note**: The dependency is already added to `pyproject.toml` but requires manual installation with CUDA flags.

For CPU-only (slower but doesn't require CUDA):

```bash
pip install llama-cpp-python
```

### 2. Install exo with llama.cpp support

The dependency is optional and marked for non-macOS platforms:

```bash
uv sync --all-packages
# or
pip install -e .
```

## Model Format

**llama.cpp requires GGUF format models**, not MLX safetensors.

### Downloading GGUF Models

GGUF models are available from HuggingFace. Popular sources:

1. **TheBloke models** - Pre-quantized GGUF models:
   - Example: `TheBloke/Llama-2-7B-Chat-GGUF`
   - Example: `TheBloke/Mistral-7B-Instruct-v0.1-GGUF`

2. **HuggingFace Hub** - Search for models with `GGUF` in the name

### Model Directory Structure

Place GGUF files in the exo models directory:

```
~/.local/share/exo/models/your--model--id/
  └── model_file.gguf
```

The model ID format uses `--` instead of `/`:
- HuggingFace ID: `TheBloke/Llama-2-7B-Chat-GGUF`
- Directory: `~/.local/share/exo/models/TheBloke--Llama-2-7B-Chat-GGUF/`
- File: `llama-2-7b-chat.Q4_K_M.gguf` (or any .gguf file)

## Usage

### Automatic Engine Selection

The engine selector automatically chooses llama.cpp when:
1. GGUF files (`.gguf`) are found in the model directory
2. llama-cpp-python is installed

Otherwise, it falls back to MLX.

### Example: Using llama.cpp Engine

1. **Download a GGUF model**:

```bash
# Example: Download a model using huggingface-cli
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF \
  llama-2-7b-chat.Q4_K_M.gguf \
  --local-dir ~/.local/share/exo/models/TheBloke--Llama-2-7B-Chat-GGUF
```

2. **Start exo**:

```bash
uv run exo
```

3. **Create instance and use the API**:

The API usage is the same as MLX - exo will automatically detect and use llama.cpp:

```bash
# List models
curl http://localhost:52415/models

# Preview instance placements
curl "http://localhost:52415/instance/previews?model_id=TheBloke--Llama-2-7B-Chat-GGUF"

# Create instance and chat (same as MLX)
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TheBloke--Llama-2-7B-Chat-GGUF",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

## Configuration

### GPU Layer Offloading

By default, llama.cpp uses all GPU layers (`n_gpu_layers=-1`). You can adjust this in `src/exo/worker/engines/llamacpp/__init__.py`:

```python
def initialize_llamacpp(
    bound_instance: BoundInstance,
    n_gpu_layers: int = -1,  # Change this value
    n_ctx: int = 4096,
    verbose: bool = False,
):
```

- `-1`: Use all GPU layers (maximum GPU usage)
- `0`: CPU only (no GPU)
- `N`: Use first N layers on GPU, rest on CPU

### Context Window Size

Adjust `n_ctx` parameter for larger context windows:

```python
n_ctx=8192  # 8K context
n_ctx=16384  # 16K context
```

**Note**: Larger context windows use more GPU memory.

## Limitations (Proof of Concept)

This is a **proof-of-concept implementation** with the following limitations:

1. **Single GPU Only**: Multi-GPU support not yet implemented
2. **No Distributed Parallelism**: Unlike MLX, doesn't support tensor/pipeline parallelism across devices
3. **Model Format**: Requires GGUF models (cannot use MLX safetensors)
4. **Manual Model Download**: GGUF models must be downloaded manually
5. **Basic Chat Template**: Chat formatting is simplified (may need improvement for specific models)

## Troubleshooting

### llama-cpp-python Import Error

If you see: `llama-cpp-python is not installed`

```bash
# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

### No GGUF Files Found

Error: `No GGUF file found in {model_dir}`

- Ensure you've downloaded a GGUF model (not MLX safetensors)
- Check that the `.gguf` file is in the correct directory
- Model directory format: `~/.local/share/exo/models/model--id/model.gguf`

### CUDA Not Available

If CUDA isn't detected, llama.cpp will fall back to CPU (slow).

- Verify CUDA is installed: `nvidia-smi`
- Ensure llama-cpp-python was compiled with CUDA: `CMAKE_ARGS="-DLLAMA_CUBLAS=on"`
- Check GPU is accessible in Docker (if using Docker)

### Out of Memory

If you get OOM errors:

1. Use a smaller quantized model (Q4 instead of Q8)
2. Reduce `n_gpu_layers` (fewer layers on GPU)
3. Reduce `n_ctx` (smaller context window)
4. Use a smaller model

## Performance Tips

1. **Use Quantized Models**: Q4_K_M or Q5_K_M provide good quality/speed tradeoff
2. **GPU Memory**: Monitor with `nvidia-smi` to ensure GPU is being used
3. **Batch Size**: llama.cpp handles requests sequentially (unlike vLLM)
4. **Context Window**: Keep context windows reasonable for your GPU memory

## Next Steps

For production use, consider:

1. Adding automatic GGUF model download support
2. Implementing multi-GPU support
3. Adding model format conversion (safetensors → GGUF)
4. Improving chat template handling for different model families
5. Adding more configuration options

## Contributing

This is a proof-of-concept. Contributions welcome for:
- Better model format handling
- Multi-GPU support
- Model download automation
- Performance optimizations

