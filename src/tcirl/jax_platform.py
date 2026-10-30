"""Where an entry point computes: a GPU entry point runs on a GPU unless its caller set `JAX_PLATFORMS=cpu`, and an entry point that needs no GPU runs JAX on the CPU."""

import os

import jax

# JAX falls back to the CPU on its own, with only a warning, when CUDA cannot start (for example under a CUDA_VISIBLE_DEVICES that names one card twice), so both rules live here and not in JAX.

CPU_PLATFORM = "cpu"


def cpu_requested() -> bool:
    """Whether the caller asked for the CPU on purpose, by setting `JAX_PLATFORMS=cpu`."""
    return os.environ.get("JAX_PLATFORMS") == CPU_PLATFORM


def require_gpu_backend(backend: str) -> None:
    """Refuse a GPU entry point's JAX backend unless it is the GPU or the caller set `JAX_PLATFORMS=cpu`."""
    if backend == "gpu" or cpu_requested():
        return
    raise RuntimeError(
        f"JAX runs on {backend!r}, not on a GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); "
        "the JAX log above names why CUDA did not start. Name one existing card once in CUDA_VISIBLE_DEVICES, "
        "or set JAX_PLATFORMS=cpu to run on the CPU on purpose."
    )


def pin_jax_to_cpu() -> None:
    """Run JAX on the CPU only, for an entry point that needs no GPU, so that it never starts CUDA; the first statement of such an entry point's `main`."""
    jax.config.update("jax_platforms", CPU_PLATFORM)
    # The update takes effect only while no backend has started, and a backend that started earlier keeps running without a word.
    backend = jax.default_backend()
    if backend != CPU_PLATFORM:
        raise RuntimeError(
            f"JAX started on {backend!r} before this entry point pinned the CPU, so a module it imports creates a JAX array at import time; build that array inside a function instead."
        )
