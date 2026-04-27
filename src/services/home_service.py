from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.domain.models import HomeStats
from src.infrastructure.vacancies_repository import VacanciesRepository


class HomeService:
    """Computes aggregate dashboard statistics from the latest vacancies dataset.

    Pure business logic: takes a repository (infrastructure), returns a domain
    entity (`HomeStats`). Has no dependency on Streamlit or on the filesystem.
    """

    _EMPTY_STATS = HomeStats(
        total_vacancies=0,
        with_salary=0,
        avg_salary=0,
        active_vacancies=0,
        last_updated="—",
    )

    def __init__(self, repository: Optional[VacanciesRepository] = None):
        self.repository = repository or VacanciesRepository()

    def get_home_statistics(self) -> HomeStats:
        df, _ = self.repository.load_latest()
        if df is None or df.empty:
            return self._EMPTY_STATS

        last_updated = self.repository.get_latest_mtime() or "—"
        return self._compute_stats(df, last_updated)

    @staticmethod
    def _compute_stats(df: pd.DataFrame, last_updated: str) -> HomeStats:
        total = len(df)

        if "salary_avg" in df.columns:
            salary_series = df["salary_avg"].dropna()
            with_salary = int(len(salary_series))
            avg_salary = int(salary_series.mean()) if with_salary else 0
        else:
            with_salary = 0
            avg_salary = 0

        if "published_at" in df.columns:
            try:
                published = pd.to_datetime(df["published_at"], errors="coerce")
                month_ago = datetime.now() - timedelta(days=30)
                active = int((published >= month_ago).sum())
            except Exception:
                active = total
        else:
            active = total

        return HomeStats(
            total_vacancies=total,
            with_salary=with_salary,
            avg_salary=avg_salary,
            active_vacancies=active,
            last_updated=last_updated,
        )
