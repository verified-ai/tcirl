"""JAX training session helpers: XLA reproducibility validation, JAX-side seeding, and reproducibility metadata persistence."""

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax

from tcirl.jax_platform import require_gpu_backend
from tcirl.logging.wandb import WandbLogger
from tcirl.session import install_sigterm_handler, require_pythonhashseed, seed_python_process

_XLA_GPU_DETERMINISTIC_OPS_FLAG = "xla_gpu_deterministic_ops"
_XLA_GPU_AUTOTUNE_LEVEL_FLAG = "xla_gpu_autotune_level"


def _xla_flag_value(xla_flags: str | None, name: str) -> str | None:
    if xla_flags is None:
        return None
    prefix = f"--{name}="
    for token in xla_flags.split():
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def has_xla_gpu_deterministic_ops(xla_flags: str | None) -> bool:
    """Check whether XLA_FLAGS enables deterministic GPU ops."""
    value = _xla_flag_value(xla_flags, _XLA_GPU_DETERMINISTIC_OPS_FLAG)
    return value is not None and value.lower() in {"1", "true"}


def has_xla_gpu_autotune_disabled(xla_flags: str | None) -> bool:
    """Check whether XLA_FLAGS sets --xla_gpu_autotune_level=0."""
    return _xla_flag_value(xla_flags, _XLA_GPU_AUTOTUNE_LEVEL_FLAG) == "0"


def validate_jax_reproducibility(
    *,
    fully_reproducible: bool,
    backend: str,
    xla_flags: str | None,
) -> tuple[bool, bool]:
    """Validate JAX determinism controls. Returns (autotune_disabled, deterministic_ops)."""

    autotune_disabled = has_xla_gpu_autotune_disabled(xla_flags)
    deterministic_ops = has_xla_gpu_deterministic_ops(xla_flags)

    if backend == "gpu" and not autotune_disabled:
        raise ValueError(
            "XLA_FLAGS must include --xla_gpu_autotune_level=0 for reproducible JAX GPU runs. "
            "The launcher scripts set this automatically; if you are invoking a "
            "training or export entry point directly, prefix with "
            "`XLA_FLAGS=--xla_gpu_autotune_level=0` or source scripts/infra/lib.sh "
            "and call ensure_xla_repro_flags (defined in scripts/infra/xla.sh)."
        )
    if fully_reproducible and backend == "gpu" and not deterministic_ops:
        raise ValueError(
            "XLA_FLAGS must include --xla_gpu_deterministic_ops=true for fully_reproducible JAX GPU runs."
        )
    return autotune_disabled, deterministic_ops


def setup_jax_reproducibility(seed: int, fully_reproducible: bool) -> dict[str, Any]:
    """Configure and report JAX reproducibility-relevant process state."""

    require_pythonhashseed(seed)
    seed_python_process(seed)

    backend = jax.default_backend()
    require_gpu_backend(backend)
    xla_flags = os.environ.get("XLA_FLAGS")
    autotune_disabled, deterministic_ops = validate_jax_reproducibility(
        fully_reproducible=fully_reproducible,
        backend=backend,
        xla_flags=xla_flags,
    )
    devices = list(jax.devices())

    return {
        "seed": seed,
        "fully_reproducible": fully_reproducible,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "backend": backend,
        "device_count": len(devices),
        "devices": [
            {
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
            }
            for device in devices
        ],
        "jax_version": jax.__version__,
        "jaxlib_version": version("jaxlib"),
        "jax_default_prng_impl": str(jax.config.values.get("jax_default_prng_impl")),
        "jax_enable_x64": bool(jax.config.values["jax_enable_x64"]),
        "jax_default_matmul_precision": str(jax.config.values.get("jax_default_matmul_precision")),
        "jax_platforms": str(jax.config.values.get("jax_platforms")),
        "xla_flags": xla_flags,
        "xla_gpu_autotune_disabled": autotune_disabled,
        "xla_gpu_deterministic_ops": deterministic_ops,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_platform_name_env": os.environ.get("JAX_PLATFORMS"),
    }


def setup_jax_run(*, seed: int, fully_reproducible: bool) -> dict[str, Any]:
    """Install the SIGTERM handler and validate/seed JAX reproducibility, before any RNG use."""
    install_sigterm_handler()
    return setup_jax_reproducibility(seed=seed, fully_reproducible=fully_reproducible)


def save_jax_reproducibility_metadata(
    run_dir: Path | None,
    logger: WandbLogger,
    metadata: dict[str, Any],
) -> None:
    """Persist JAX reproducibility metadata alongside the run outputs.

    `run_dir` is `WandbLogger.run_dir`; the file write is skipped when it is
    `None` (no run open), since there is then nowhere for run artifacts to go.
    Summary logs remain (they are no-ops when no run is open).
    """

    if run_dir is not None:
        metadata_path = run_dir / "reproducibility.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        logger.save_file(metadata_path, base_path=run_dir)

    logger.log_summary("repro/backend", metadata["backend"])
    logger.log_summary("repro/device_count", int(metadata["device_count"]))
    logger.log_summary("repro/fully_reproducible", int(metadata["fully_reproducible"]))
    logger.log_summary("repro/jax_default_prng_impl", str(metadata["jax_default_prng_impl"]))
    logger.log_summary(
        "repro/jax_default_matmul_precision",
        str(metadata["jax_default_matmul_precision"]),
    )
    logger.log_summary("repro/pythonhashseed", str(metadata["pythonhashseed"]))
    logger.log_summary(
        "repro/xla_gpu_autotune_disabled",
        int(bool(metadata["xla_gpu_autotune_disabled"])),
    )
    logger.log_summary(
        "repro/xla_gpu_deterministic_ops",
        int(bool(metadata["xla_gpu_deterministic_ops"])),
    )
