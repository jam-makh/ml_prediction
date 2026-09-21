"""Logging for every entry point: one console sink and one file per run.

Everything the scripts used to ``print`` goes through loguru instead, so a run
leaves a timestamped file behind rather than only scrollback. The console sink
prints the bare message -- the output is mostly tables, and a timestamp prefix
on every line of a table makes it unreadable. The file sink carries time, level
and source line, because that is where a run is read back afterwards.

Call ``setup_logging`` once, from the entry point. Library code only does
``from loguru import logger`` and logs.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.config import PROJECT_ROOT

DEFAULT_LOG_DIR = "results/logs"
CONSOLE_FORMAT = "<level>{message}</level>"
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{line} | {message}"

# Id of the current console sink, so ``console_level`` can swap it without
# touching the file sink.
_console_sink: int | None = None


def setup_logging(
    config: dict[str, Any] | None = None, run_name: str = "run"
) -> Path:
    """Replace loguru's default sink with a console sink and a per-run file.

    Parameters
    ----------
    config : dict, optional
        Parsed config. ``output.log_dir`` sets the directory, relative to the
        project root. Default ``results/logs``.
    run_name : str, optional
        Prefix of the log file name, usually the entry point. Default ``run``.

    Returns
    -------
    pathlib.Path
        The log file this run writes to.
    """
    global _console_sink
    log_dir = PROJECT_ROOT / str(
        ((config or {}).get("output") or {}).get("log_dir", DEFAULT_LOG_DIR)
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    _console_sink = logger.add(sys.stderr, level="INFO", format=CONSOLE_FORMAT)
    path = log_dir / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.add(path, level="DEBUG", format=FILE_FORMAT, encoding="utf-8")
    return path


@contextmanager
def console_level(level: str) -> Iterator[None]:
    """Raise the console threshold for a block; the file still gets everything.

    What ``pipeline.py``'s quiet mode uses instead of redirecting stdout: the
    underlying scripts' output stays out of the terminal but is never lost,
    since the file sink is untouched.

    Parameters
    ----------
    level : str
        Minimum level shown on the console inside the block, e.g. ``WARNING``.

    Yields
    ------
    None
    """
    global _console_sink
    if _console_sink is None:
        yield
        return
    logger.remove(_console_sink)
    _console_sink = logger.add(sys.stderr, level=level, format=CONSOLE_FORMAT)
    try:
        yield
    finally:
        logger.remove(_console_sink)
        _console_sink = logger.add(sys.stderr, level="INFO", format=CONSOLE_FORMAT)
