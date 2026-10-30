#!/bin/bash
# Verify JAX sees a GPU before a reproduce pipeline spends hours on the wrong device.
#
# Usage: check_accelerator.sh
#
# The same backend policy serves the launchers and the training commands.
# JAX can fall back to the CPU with only a warning, so an accidental CPU backend fails here.
# An explicit JAX_PLATFORMS=cpu request uses the pool's existing CPU path.
#
# Fix a failure with `uv sync --extra cuda`.
set -euo pipefail

if ! uv run python -c '
import jax
from tcirl.jax_platform import require_gpu_backend

require_gpu_backend(jax.default_backend())
devices = jax.devices()
print(f"  accelerator: {devices}")
'; then
    exit 1
fi
