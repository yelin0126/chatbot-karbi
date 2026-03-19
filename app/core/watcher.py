"""
Background file watcher — monitors data/ folder and auto-ingests new files.

Drop a PDF or image into the data/ folder from anywhere (file manager,
scp, cp command) and it gets automatically parsed, chunked, and stored
in ChromaDB. No API call needed.
"""

import time
import threading
import logging
from pathlib import Path
from typing import Set

from app.config import DATA_DIR

logger = logging.getLogger("tilon.watcher")

WATCH_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
POLL_INTERVAL = 5  # seconds


class FileWatcher:
    """Watches data/ folder for new files and auto-ingests them."""

    def __init__(self):
        self._known_files: Set[str] = set()
        self._thread: threading.Thread = None
        self._running = False

    def start(self):
        """Start watching in a background thread."""
        self._running = True
        self._known_files = self._scan_existing()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(
            "File watcher started — monitoring %s (%d existing files)",
            DATA_DIR, len(self._known_files),
        )

    def stop(self):
        """Stop the watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("File watcher stopped.")

    def _scan_existing(self) -> Set[str]:
        """Get set of files currently in data/."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        files = set()
        for ext in WATCH_EXTENSIONS:
            for f in DATA_DIR.glob(f"*{ext}"):
                files.add(str(f))
        return files

    def _watch_loop(self):
        """Poll for new files every POLL_INTERVAL seconds."""
        while self._running:
            try:
                current_files = self._scan_existing()
                new_files = current_files - self._known_files

                for file_path in sorted(new_files):
                    self._ingest_file(Path(file_path))

                self._known_files = current_files

            except Exception as e:
                logger.error("File watcher error: %s", e)

            time.sleep(POLL_INTERVAL)

    def _ingest_file(self, file_path: Path):
        """Ingest a single new file."""
        # Wait a moment for file to finish writing
        time.sleep(1)

        try:
            # Import here to avoid circular imports
            from app.pipeline.ingest import ingest_single_file

            logger.info("Auto-ingesting new file: %s", file_path.name)
            result = ingest_single_file(file_path)
            logger.info(
                "Auto-ingest complete: %s → %d chunks",
                file_path.name, result.get("count", 0),
            )
        except Exception as e:
            logger.error("Auto-ingest failed for %s: %s", file_path.name, e)


# Global watcher instance
_watcher = FileWatcher()


def start_watcher():
    _watcher.start()


def stop_watcher():
    _watcher.stop()