#!/bin/bash
set -e

# Ensure we're using the correct Python and paths
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Execute the command passed as arguments using uv run
# uv run will handle the virtual environment automatically
exec uv run "$@"

