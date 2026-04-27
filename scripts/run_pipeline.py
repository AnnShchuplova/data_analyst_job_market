"""End-to-end pipeline: collect -> filter -> clean.

Designed as the single entrypoint for the parser container (called by cron).
Produces a fresh `data/processed/new_cleaned_vacancies_*.csv` on every run.

Run from the project root:
    python -m scripts.run_pipeline
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when invoked via `python -m scripts.run_pipeline`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_data import collect_raw_data  # noqa: E402
from scripts.filter_data import filter_and_save  # noqa: E402
from src.data_processing.cleaner import DataCleaner  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("pipeline")


def main() -> int:
    # All downstream scripts use paths relative to CWD — anchor to project root.
    os.chdir(PROJECT_ROOT)

    try:
        log.info("Step 1/3: collecting raw data from hh.ru")
        _, raw_path = collect_raw_data()
        log.info("Raw data saved: %s", raw_path)

        log.info("Step 2/3: filtering analyst vacancies")
        _, filtered_path = filter_and_save(raw_path)
        log.info("Filtered data saved: %s", filtered_path)

        log.info("Step 3/3: cleaning and feature extraction")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = PROJECT_ROOT / "data" / "processed" / f"new_cleaned_vacancies_{timestamp}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        cleaner = DataCleaner()
        cleaner.run_full_clean(
            input_file=filtered_path,
            output_file=str(output_csv),
            remove_outliers=False,
        )
        log.info("Clean CSV saved: %s", output_csv)

        log.info("Pipeline completed successfully")
        return 0
    except Exception:
        log.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
