import glob
import hashlib
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class VacanciesRepository:
    """Data-access gateway for the latest cleaned vacancies CSV.

    Lives in the infrastructure layer: hides filesystem, CSV format, and
    checksumming from services/views. Services receive a ready-to-use DataFrame
    and metadata; they never touch `os.path` directly.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Project root is three levels up from this file: src/infrastructure/<this>.
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            data_dir = os.path.join(project_root, "finaldata")
        self.data_dir = data_dir

    def _find_latest_path(self) -> Optional[str]:
        if not os.path.exists(self.data_dir):
            return None
        files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        return max(files, key=os.path.getmtime) if files else None

    def load_latest(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Return (dataframe, filename) for the newest CSV, or (None, None)."""
        path = self._find_latest_path()
        if not path:
            return None, None
        try:
            df = pd.read_csv(path, encoding="utf-8")
            return df, os.path.basename(path)
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return None, None

    def compute_checksum(self) -> str:
        """MD5 of the latest CSV bytes. Used as a cache key for services."""
        path = self._find_latest_path()
        if not path:
            return ""
        h = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as exc:
            logger.error("Cannot read file for checksum: %s", exc)
            return ""

    def get_latest_mtime(self) -> Optional[str]:
        """Formatted modification date of the latest CSV (dd.mm.yyyy)."""
        path = self._find_latest_path()
        if not path:
            return None
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d.%m.%Y")
