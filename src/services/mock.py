import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import List

from src.domain.models import (
    HomeStats,
    ClusterEntity,
    AggregationStats,
    AggregationCharts,
    SalaryPredictionResult
)


class MockService:
    def get_home_statistics(self) -> HomeStats:
        return HomeStats(
            total_vacancies=15430,
            with_salary=12300,
            avg_salary=145000,
            active_vacancies=3420,
            last_updated=datetime.now().strftime('%d.%m.%Y')
        )

    def get_aggregation_stats(self) -> AggregationStats:
        return AggregationStats(
            avg_salary="145 000 ₽",
            total_vacancies=15430,
            yearly_change="+12%"
        )

    def get_aggregation_charts(self) -> AggregationCharts:
        df1 = pd.DataFrame(
            np.random.randn(20, 3).cumsum(axis=0) + 100,
            columns=["Junior", "Middle", "Senior"]
        )
        df2 = pd.DataFrame(
            np.random.randint(20, 100, size=(5, 1)),
            index=["Python", "SQL", "Docker", "Git", "Pandas"],
            columns=["Demand"]
        )
        return AggregationCharts(trend_chart=df1, skills_chart=df2)

    def get_clusters(self) -> List[ClusterEntity]:
        return [
            ClusterEntity(1, "ML Engineers", "Разработка моделей", 847, "180K ₽", ["Python", "PyTorch"]),
            ClusterEntity(2, "Data Analysts", "Анализ данных", 1234, "110K ₽", ["SQL", "Tableau"]),
            ClusterEntity(3, "Data Engineers", "ETL пайплайны", 950, "160K ₽", ["Spark", "Airflow"]),
        ]

    def predict_salary(self, job_title: str, exp: int, skills: List[str]) -> SalaryPredictionResult:
        time.sleep(0.5)

        base_salary = 60000 + (exp * 15000)
        final_salary = round(base_salary, -3)

        dist_chart = pd.DataFrame(
            np.random.normal(final_salary, 15000, 100),
            columns=["salary_dist"]
        )

        return SalaryPredictionResult(
            predicted_salary=final_salary,
            currency="₽",
            confidence_interval=(final_salary - 10000, final_salary + 10000),
            market_comparison_chart=dist_chart
        )