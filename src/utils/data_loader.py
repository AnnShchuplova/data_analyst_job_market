import hashlib
import os
import sys
import glob
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _find_latest_csv() -> Optional[str]:
    """Returns path to the latest new_cleaned_vacancies*.csv, or None if not found."""
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = glob.glob(os.path.join(base_dir, 'data', 'processed', "new_cleaned_vacancies*.csv"))
    return max(files, key=os.path.getmtime) if files else None


def compute_data_checksum() -> str:
    """MD5 of the raw bytes of the latest vacancies CSV. Returns '' if no file found."""
    path = _find_latest_csv()
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


def simplify_job_name(name):
    if not isinstance(name, str):
        return 'Other'

    name_lower = name.lower()

    categories = {
        'BI Analyst': ['bi', 'tableau', 'power bi', 'qlik', 'looker'],
        'System Analyst': ['system', 'системн'],
        'Business Analyst': ['business', 'бизнес'],
        'Financial Analyst': ['financial', 'финанс'],
        'Data Analyst': ['data', 'данны', 'product', 'продуктов'],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name_lower:
                if keyword == 'bi' and ('big data' in name_lower or 'mobile' in name_lower):
                    continue
                return category

    return 'Other'


def load_vacancies_data():
    latest_file = _find_latest_csv()

    if not latest_file:
        logger.error("Files not found in data/processed/")
        return pd.DataFrame()

    try:
        df = pd.read_csv(latest_file)

        useful_cols = [
            'id',
            'name',
            'salary_avg',
            'avg_experience_years',
            'min_experience_years',
            'requirement',
            'responsibility',
            'schedule_name',
            'employment_name',
            'employer_name',
            'area_name',
            'skills_list'
        ]

        df_ml = df[useful_cols].copy()
        df_ml['name'] = df_ml['name'].apply(simplify_job_name)

        return df_ml

    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    df = load_vacancies_data()
