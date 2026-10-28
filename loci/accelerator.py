"""Hardware accelerator detection and array backend selection.

The backend is cupy (CUDA) when it imports and runs, else numpy. The fallback is never silent: the chosen
backend and the reason for any fallback are logged once (INFO for a deliberate or expected choice, WARNING for
an unexpected cupy failure) and reported by ``accelerator_info()`` / ``accelerator_health()``.

A cupy *import* failure is permanent for the process (cupy is not installed). Any other failure (driver busy,
out of memory, a transient CUDA error) is retried with backoff from ``get_array_module()`` instead of
disabling the GPU until restart.
"""
import logging
import os
import time

_LOG = logging.getLogger("loci.accelerator")

_RETRY_INITIAL_S = 30.0
_RETRY_MAX_S = 3600.0

_monotonic = time.monotonic  # injectable clock for the retry backoff

_BACKEND = None
_XP = None
_REASON = ""          # why the backend is cpu ("" when it is cuda)
_FALLBACK = False     # True when cpu was not asked for (cupy missing or failing)
_PERMANENT = False    # True when retrying cannot help (override, or cupy not importable)
_RETRY_AT = None      # monotonic time of the next cupy retry
_RETRY_DELAY_S = _RETRY_INITIAL_S
_RETRIES = 0


def _try_cupy():
    """Return (cupy, None) when it works, else (None, exception)."""
    try:
        import cupy as cp
        cp.array([1.0])
        return cp, None
    except Exception as exc:  # noqa: BLE001 - classified by the caller
        return None, exc


def _use_numpy(reason: str, fallback: bool, permanent: bool, level: int) -> None:
    global _BACKEND, _XP, _REASON, _FALLBACK, _PERMANENT, _RETRY_AT
    import numpy as np
    changed = _BACKEND != "cpu" or _REASON != reason
    _XP = np
    _BACKEND = "cpu"
    _REASON = reason
    _FALLBACK = fallback
    _PERMANENT = permanent
    _RETRY_AT = None if permanent else _monotonic() + _RETRY_DELAY_S
    if changed:
        _LOG.log(level, "accelerator backend=cpu (numpy): %s", reason)


def _detect():
    global _BACKEND, _XP, _REASON, _FALLBACK, _PERMANENT, _RETRY_AT
    override = os.getenv("LOCI_ACCELERATOR", "").lower()
    if override == "cpu":
        _use_numpy("LOCI_ACCELERATOR=cpu", fallback=False, permanent=True, level=logging.INFO)
        return
    cp, exc = _try_cupy()
    if cp is not None:
        recovered = _FALLBACK
        _XP, _BACKEND, _REASON, _FALLBACK, _PERMANENT, _RETRY_AT = cp, "cuda", "", False, False, None
        _LOG.info("accelerator backend=cuda (cupy)%s", " (recovered after retry)" if recovered else "")
        return
    if isinstance(exc, ImportError):
        _use_numpy(f"cupy not importable: {exc!r}", fallback=True, permanent=True, level=logging.INFO)
    else:
        _use_numpy(f"cupy failed: {exc!r}; retrying with backoff", fallback=True, permanent=False,
                   level=logging.WARNING)


_detect()


def _maybe_retry() -> None:
    """Retry cupy after a transient failure, doubling the delay up to _RETRY_MAX_S."""
    global _RETRY_AT, _RETRY_DELAY_S, _RETRIES
    if _PERMANENT or _RETRY_AT is None or _monotonic() < _RETRY_AT:
        return
    _RETRIES += 1
    _RETRY_DELAY_S = min(_RETRY_DELAY_S * 2, _RETRY_MAX_S)
    _detect()


def get_array_module():
    _maybe_retry()
    return _XP


def get_backend() -> str:
    return _BACKEND


def accelerator_info() -> dict:
    """Backend report: ``accelerator`` ("cuda"/"cpu"), ``backend`` (module name), ``device_count`` (CUDA
    devices cupy sees; 0 on cpu), ``fallback`` (True when cpu was not asked for), ``reason`` (why cpu,
    "" on cuda) and ``retries`` (cupy retry attempts after a transient failure)."""
    info = {
        "accelerator": _BACKEND,
        "backend": _XP.__name__,
        "device_count": 0,
        "fallback": _FALLBACK,
        "reason": _REASON,
        "retries": _RETRIES,
    }
    if _BACKEND == "cuda":
        try:
            import cupy
            info["device_count"] = cupy.cuda.runtime.getDeviceCount()
        except Exception as e:  # noqa: BLE001
            info["device_count"] = None
            info["cuda_error"] = str(e)
    return info


def accelerator_health() -> dict:
    """Hook-ready dict for loci_health. Importing this module runs the cupy probe (it loads CUDA), so the
    hook should call it only when ``"loci.accelerator" in sys.modules`` -- ``loci.drpg`` imports it lazily
    and a server that never scored a large batch has not initialised it."""
    return {"accelerator": accelerator_info()}
