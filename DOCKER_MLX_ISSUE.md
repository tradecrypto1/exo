# MLX on Linux/Docker - Known Issue

## Problem

MLX fails to load on Linux with the error:
```
ImportError: libmlx.so: cannot open shared object file: No such file or directory
```

## Root Cause

MLX (Apple's ML framework) is primarily designed for Apple Silicon (macOS). While MLX has experimental Linux CPU support, the native libraries (`libmlx.so`) may not be properly available in the Docker container.

## Current Status

- **MLX on macOS**: ✅ Fully supported (GPU acceleration)
- **MLX on Linux**: ⚠️ Experimental CPU support, may not work in Docker
- **llama.cpp**: ✅ Works on Linux with CUDA/CPU

## Solutions

### Option 1: Use GGUF Models with llama.cpp (Recommended for Docker)

Instead of MLX safetensors models, use GGUF models:

1. **Download a GGUF model**:
   ```bash
   # Example: Download Qwen3 in GGUF format
   huggingface-cli download TheBloke/Qwen3-0.6B-Instruct-GGUF \
     qwen3-0.6b-instruct.Q4_K_M.gguf \
     --local-dir ~/.local/share/exo/models/TheBloke--Qwen3-0.6B-Instruct-GGUF
   ```

2. **Use the model ID** with `--` instead of `/`:
   ```
   TheBloke--Qwen3-0.6B-Instruct-GGUF
   ```

3. **exo will automatically detect GGUF files** and use llama.cpp engine

### Option 2: Run on macOS

MLX works natively on macOS with Apple Silicon. If you have access to a Mac, run exo there.

### Option 3: Wait for Official MLX Linux Support

According to the exo roadmap (`PLATFORMS.md`), Linux CUDA support is planned but not yet implemented.

## Error Message

When you try to use an MLX model (safetensors) on Linux, you'll see:

```
MLX engine is not available on this platform (Linux/Docker).
Model 'mlx-community/Qwen3-0.6B-4bit' requires MLX format (safetensors).

Options:
1. Use a GGUF model instead (download GGUF version to ...)
2. Run on macOS where MLX is supported
3. Wait for MLX Linux CUDA support (currently in development)
```

## Finding GGUF Alternatives

Many models have GGUF versions available on HuggingFace:

- Search for: `{model-name} GGUF`
- Popular sources: `TheBloke/*-GGUF` repositories
- Example: `TheBloke/Qwen3-0.6B-Instruct-GGUF`

## Quick Fix

For immediate use in Docker, switch to a GGUF model:

```bash
# In the container or on host
huggingface-cli download TheBloke/Qwen3-0.6B-Instruct-GGUF \
  qwen3-0.6b-instruct.Q4_K_M.gguf \
  --local-dir /root/.local/share/exo/models/TheBloke--Qwen3-0.6B-Instruct-GGUF
```

Then use model ID: `TheBloke--Qwen3-0.6B-Instruct-GGUF` in the API.

