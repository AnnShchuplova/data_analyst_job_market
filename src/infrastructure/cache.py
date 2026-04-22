import hashlib
import logging
from pathlib import Path
from typing import Optional

import joblib

from src.domain.models import ClusteringResult

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


class ClusteringCache:
    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, data_checksum: str, features: list, k_range: range) -> Optional[ClusteringResult]:
        path = self._path(self._make_key(data_checksum, features, k_range))
        if not path.exists():
            return None
        try:
            result = joblib.load(path)
            if not isinstance(result, ClusteringResult):
                raise TypeError(f"Unexpected type in cache: {type(result)}")
            return result
        except Exception as exc:
            logger.warning("Corrupted cache entry %s — treating as miss. %s", path.name, exc)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, data_checksum: str, features: list, k_range: range, result: ClusteringResult) -> None:
        path = self._path(self._make_key(data_checksum, features, k_range))
        try:
            joblib.dump(result, path, compress=3)
        except Exception as exc:
            logger.warning("Failed to write cache entry: %s", exc)

    def _make_key(self, data_checksum: str, features: list, k_range: range) -> str:
        raw = f"{data_checksum}|{','.join(sorted(features))}|{k_range.start}-{k_range.stop}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"clustering_{key}.pkl"
