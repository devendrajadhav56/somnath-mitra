"""Centralised logging setup for Somnath Mitra backend.

Uses a ContextVar to carry a per-request ID through the async call stack,
so every log line from the same request shares the same req_id tag.

Usage in services:
    from services.applog import log_step
    log_step("intent", tools="plan_route_to_somnath", use_rag=False, ms=980)

Usage in middleware / routers:
    from services.applog import set_req_id, new_req_id
    set_req_id(new_req_id())
"""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config

# ── Request ID context ────────────────────────────────────────────────────────

_req_id_var: ContextVar[str] = ContextVar("req_id", default="-")


def new_req_id() -> str:
    return uuid.uuid4().hex[:8]


def set_req_id(req_id: str) -> None:
    _req_id_var.set(req_id)


def get_req_id() -> str:
    return _req_id_var.get()


# ── Filter that injects req_id into every LogRecord ──────────────────────────

class _ReqIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.req_id = _req_id_var.get()
        return True


# ── Logger setup ─────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "somnath_mitra.log"

_logger = logging.getLogger("somnath_mitra")


def setup_logging(level: int | None = None) -> None:
    """
    Configure file-only logging. Call once at application startup.
    Rotating at 10 MB, keeping 5 backups.

    Level defaults to config.LOG_LEVEL (INFO). Set LOG_LEVEL=DEBUG in the
    environment to also capture full payloads logged via log_debug().
    """
    if level is None:
        level = getattr(logging, config.LOG_LEVEL, logging.INFO)

    LOG_DIR.mkdir(exist_ok=True)

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.addFilter(_ReqIdFilter())
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s req=%(req_id)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    _logger.setLevel(level)
    _logger.addHandler(handler)
    _logger.propagate = False  # don't bubble up to root → nothing hits terminal


# ── Public helpers ────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a child of the `somnath_mitra` logger for use in any module.

    Child loggers propagate to the `somnath_mitra` parent, so they inherit its
    handler, the per-request `req_id` filter, and its level — every line lands
    in the same file, tagged with the request ID, and respects LOG_LEVEL.

    Usage:
        from services.applog import get_logger
        log = get_logger(__name__)
        log.warning("intent failed: %s", exc)   # tagged with req_id
        log.debug("full doc: %s", doc)           # silent unless level is DEBUG
    """
    return _logger.getChild(name)


def log_step(step: str, **kwargs) -> None:
    """Emit one INFO line: `step=<step> key=value key=value ...`"""
    parts = [f"step={step}"] + [f"{k}={v!r}" if " " in str(v) else f"{k}={v}" for k, v in kwargs.items()]
    _logger.info(" ".join(parts))


def log_error(step: str, exc: Exception, **kwargs) -> None:
    """Emit one ERROR line with exception info."""
    parts = [f"step={step}", f"error={type(exc).__name__}:{exc}"] + [f"{k}={v}" for k, v in kwargs.items()]
    _logger.error(" ".join(parts))


def log_debug(step: str, **kwargs) -> None:
    """Emit one DEBUG line with full payloads (chunks, tool output, prompts).

    Silent unless LOG_LEVEL=DEBUG. The isEnabledFor guard avoids building large
    payload strings when debug logging is off.
    """
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    parts = [f"step={step}"] + [f"{k}={v!r}" for k, v in kwargs.items()]
    _logger.debug(" ".join(parts))
