"""Robust file-write helpers for training telemetry.

Windows has a long tail of transient permission failures when a writer and a
reader briefly race on the same file:

* ``os.replace`` fails with ``PermissionError`` if another process has the
  destination open for reading at the exact moment of the swap.
* A plain ``write_text`` on an existing file fails if an antivirus scanner or
  a GUI tail-reader is holding the handle.

The helpers here wrap every training-side JSON write with:

* a per-PID temp file (avoids temp-name collisions across concurrent trainings);
* a retry loop with linear backoff (absorbs brief AV / GUI-read locks);
* a graceful in-place fallback (atomicity is nice-to-have; never crashing the
  training job is mandatory);
* a boolean return so the caller can warn and keep going instead of raising.

Do not replace this with a simple ``json.dump`` -- the older, naive writer is
exactly what killed the overnight self-play run on 2026-04-19 after 51k steps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


_WRITE_RETRIES = 20
_WRITE_RETRY_BASE_SECONDS = 0.02
_WRITE_RETRY_CAP_SECONDS = 0.10


def _retry_sleep(attempt: int) -> None:
    time.sleep(min(_WRITE_RETRY_BASE_SECONDS * (attempt + 1), _WRITE_RETRY_CAP_SECONDS))


def _temp_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = 2) -> bool:
    """Serialize ``payload`` as JSON and write it to ``path`` without crashing.

    Returns ``True`` on success. On failure returns ``False`` -- callers should
    treat telemetry writes as best-effort and keep the training loop running.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=True, indent=indent, default=str)
    temp_path = _temp_sibling(path)

    # Attempt 1: atomic write via temp + os.replace.
    for attempt in range(_WRITE_RETRIES):
        try:
            temp_path.write_text(serialized, encoding="utf-8")
            os.replace(temp_path, path)
            return True
        except (PermissionError, OSError):
            if attempt == _WRITE_RETRIES - 1:
                break
            _retry_sleep(attempt)

    # Attempt 2: in-place overwrite. Not atomic, but keeps training alive.
    for attempt in range(_WRITE_RETRIES):
        try:
            path.write_text(serialized, encoding="utf-8")
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return True
        except (PermissionError, OSError):
            if attempt == _WRITE_RETRIES - 1:
                break
            _retry_sleep(attempt)

    try:
        temp_path.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def append_jsonl(path: Path, payload: Any) -> bool:
    """Append one JSON line to ``path`` with the same never-crash guarantees."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=True, default=str) + "\n"
    for attempt in range(_WRITE_RETRIES):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
            return True
        except (PermissionError, OSError):
            if attempt == _WRITE_RETRIES - 1:
                return False
            _retry_sleep(attempt)
    return False
