# Dockerfile for exo on Linux (CPU mode)
# Note: MLX on Linux runs on CPU only

FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.13 from deadsnakes PPA (Ubuntu 24.04 doesn't have 3.13 by default)
# Note: python3.13-distutils is not available (distutils was removed in Python 3.12+)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python3 -> python3.13
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python3.13-config

# Install Node.js (LTS version)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Rust (nightly toolchain as required)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN rustup toolchain install nightly && \
    rustup default nightly

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy Python dependency files first (better caching)
COPY pyproject.toml uv.lock README.md ./

# Copy Rust toolchain config (rustup looks for it in project root)
COPY rust/rust-toolchain.toml rust-toolchain.toml
COPY Cargo.toml Cargo.lock ./

# Copy Rust source code structure (needed for maturin build)
COPY rust/ rust/

# Copy Python source code
COPY src/ src/

# Install Python dependencies using uv
# uv will use maturin to build exo_pyo3_bindings automatically
RUN uv sync --all-packages

# Copy dashboard files
COPY dashboard/ dashboard/

# Build dashboard
RUN cd dashboard && \
    npm install && \
    npm run build

# Expose the exo API port
EXPOSE 52415

# Set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default command
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["exo"]

