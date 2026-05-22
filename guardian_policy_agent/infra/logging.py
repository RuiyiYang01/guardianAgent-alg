from __future__ import annotations
import logging, json, os, sys
from datetime import datetime
from typing import Any, Dict

def _json_formatter(record: logging.LogRecord) -> str:
    """
    Minimal JSON formatter (no external deps). Includes ISO8601 timestamp.
    """
    payload: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "lvl": record.levelname,
        "name": record.name,
        "msg": record.getMessage(),
    }
    if record.exc_info:
        payload["exc_info"] = True
    return json.dumps(payload, ensure_ascii=False)

class JsonLogHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = _json_formatter(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging() -> None:
    """
    Configure root logger. Controlled by ENV:
      LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
      LOG_JSON=true|false
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    as_json = os.getenv("LOG_JSON", "false").lower() in {"1","true","yes","on"}

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (uvicorn may pre-configure)
    for h in list(root.handlers):
        root.removeHandler(h)

    if as_json:
        handler = JsonLogHandler(stream=sys.stdout)
        root.addHandler(handler)
    else:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        root.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
