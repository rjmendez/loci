"""Hardware accelerator detection and array backend selection."""
import os

_BACKEND = None
_XP = None


def _detect():
    global _BACKEND, _XP
    override = os.getenv("LOCI_ACCELERATOR", "").lower()
    if override != "cpu":
        try:
            import cupy as cp
            cp.array([1.0])
            _XP = cp
            _BACKEND = "cuda"
            return
        except Exception:
            pass
    import numpy as np
    _XP = np
    _BACKEND = "cpu"


_detect()


def get_array_module():
    return _XP


def get_backend() -> str:
    return _BACKEND


def accelerator_info() -> dict:
    info = {"accelerator": _BACKEND, "backend": _XP.__name__}
    if _BACKEND == "cuda":
        try:
            import cupy
            info["device_count"] = cupy.cuda.runtime.getDeviceCount()
        except Exception as e:
            info["cuda_error"] = str(e)
    return info
