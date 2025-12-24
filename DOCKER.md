# Running exo on Windows with Docker

This guide explains how to run exo in a Docker container on Windows.

## Prerequisites

1. **Docker Desktop for Windows** - Install from [docker.com](https://docs.docker.com/desktop/install/windows-install/)
2. **WSL 2** (recommended) - Docker Desktop uses WSL 2 backend for better performance

## Important Notes

- **MLX on Linux**: exo uses MLX which currently runs on **CPU only** on Linux. GPU acceleration is **not yet available**.
- **GPU Support Status**: According to the project roadmap, Linux CUDA support is planned but not yet implemented. MLX is primarily designed for Apple Silicon (Metal), and Linux CUDA support is still in development.
- **Performance**: CPU inference will be significantly slower than GPU-accelerated inference on macOS.
- **Multi-device clusters**: Docker networking can be configured for multi-container setups, but single-container deployment is typical.

### Why No GPU Support?

- MLX (Apple's ML framework) is designed for Apple Silicon's Metal framework
- Linux CUDA support is marked as "Planned" in the project roadmap but not yet implemented
- Even with NVIDIA GPU passthrough in Docker, MLX won't use it until CUDA support is added
- The current setup uses MLX's CPU fallback mode

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
```

Or manually:

```bash
docker build -t exo:latest .
```

### 2. Run exo

```bash
docker-compose up
```

Or manually:

```bash
docker run -d \
  --name exo \
  -p 52415:52415 \
  exo:latest
```

### 3. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:52415
```

## Configuration

### Port Configuration

The default port is `52415`. To change it, modify `docker-compose.yml`:

```yaml
ports:
  - "YOUR_PORT:52415"
```

### Volume Mounts

The `docker-compose.yml` includes:
- **exo-data**: Persistent storage for exo state and configuration
- **models**: Optional mount for HuggingFace models cache (uncomment if needed)

### Environment Variables

You can set environment variables in `docker-compose.yml`:

```yaml
environment:
  - RUST_LOG=info
  - EXO_DEBUG=1
```

## Building from Source

The Dockerfile builds everything from source:
1. Installs Python 3.13, Node.js, Rust (nightly)
2. Builds Rust bindings
3. Installs Python dependencies via `uv`
4. Builds the dashboard

Build time can take 10-20 minutes depending on your system.

## Troubleshooting

### Port Already in Use

If port 52415 is already in use:

```bash
# Check what's using the port
netstat -ano | findstr :52415

# Or change the port in docker-compose.yml
```

### Container Won't Start

Check logs:

```bash
docker-compose logs
# or
docker logs exo
```

### Performance Issues

- CPU inference is inherently slower than GPU
- Consider allocating more CPU cores in Docker Desktop settings
- Check Docker Desktop resource allocation (Settings → Resources)

### Model Downloads

Models are downloaded to the container. To persist them:

1. Uncomment the models volume in `docker-compose.yml`
2. Models will be cached in `./models` on your host

## Advanced Usage

### Run Commands in Container

```bash
docker exec -it exo bash
```

### View Real-time Logs

```bash
docker-compose logs -f
```

### Rebuild After Code Changes

```bash
docker-compose build --no-cache
docker-compose up
```

### Multiple Instances (Cluster Simulation)

To run multiple exo instances for testing:

```yaml
# docker-compose.cluster.yml
version: '3.8'

services:
  exo1:
    build: .
    ports:
      - "52415:52415"
    # ... other config

  exo2:
    build: .
    ports:
      - "52416:52415"
    # ... other config
```

## Limitations

1. **No GPU Support**: MLX GPU acceleration on Linux is not yet implemented. Currently only CPU mode works on Linux/Docker. GPU support requires macOS (Apple Silicon) or future Linux CUDA support.
2. **Performance**: CPU inference is significantly slower than GPU inference
3. **Networking**: Container networking may affect device discovery (use host network mode if needed)

## Future GPU Support

When MLX CUDA support becomes available for Linux:

1. **Install NVIDIA Container Toolkit** (on Linux host):
   ```bash
   # On Linux host (not Windows)
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Use GPU-enabled compose file**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
   ```

3. **For Windows/WSL2**: GPU passthrough requires WSL2 with NVIDIA drivers, but MLX CUDA support must be implemented first.

## API Usage

Once running, use the API as described in the main README:

```bash
# List models
curl http://localhost:52415/models

# Get instance previews
curl "http://localhost:52415/instance/previews?model_id=llama-3.2-1b"

# Create instance and chat (see README for full examples)
curl -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Support

For issues specific to Docker/Windows, check:
- [exo GitHub Issues](https://github.com/exo-explore/exo/issues)
- Docker Desktop logs
- Container logs: `docker logs exo`

