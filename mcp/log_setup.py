"""Optional rotating file log for the Loci server.

The server logs to stderr only, so a hand-started or scheduled instance leaves no record
once its console is gone. Set LOCI_LOG_FILE to also write a size-rotated log.
"""
import logging
import logging.handlers
import os

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def install_file_logging(environ=None, root=None):
    """Attach a RotatingFileHandler to the root logger when LOCI_LOG_FILE is set.

    Returns the handler, or None when disabled or the file cannot be opened (never raises:
    logging problems must not stop the server). Idempotent for the same path.
    """
    env = os.environ if environ is None else environ
    path = (env.get("LOCI_LOG_FILE") or "").strip()
    if not path:
        return None
    root = root or logging.getLogger()
    target = os.path.abspath(path)
    for h in root.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == target:
            return h
    try:
        max_bytes = max(1024, int(env.get("LOCI_LOG_MAX_BYTES") or 10 * 1024 * 1024))
        backups = max(0, int(env.get("LOCI_LOG_BACKUPS") or 5))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(target, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    except (OSError, ValueError) as exc:
        logging.getLogger("loci-mcp").warning("LOCI_LOG_FILE=%s not usable (%s); logging to stderr only", path, exc)
        return None
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    return handler
