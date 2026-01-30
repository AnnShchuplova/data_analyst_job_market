from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pandas as pd

@dataclass
class HomeStats:
    total_vacancies: int
    with_salary: int
    avg_salary: int
    active_vacancies: int
    last_updated: str

@dataclass
class AggregationStats:
    avg_salary: str
    total_vacancies: int
    yearly_change: str

@dataclass
class AggregationCharts:
    trend_chart: pd.DataFrame
    skills_chart: pd.DataFrame

@dataclass
class ClusterEntity:
    id: int
    title: str
    description: str
    vacancies_count: int
    avg_salary: str
    skills: List[str]

@dataclass
class SalaryPredictionResult:
    predicted_salary: int
    currency: str
    confidence_interval: Tuple[int, int]
    market_comparison_chart: pd.DataFrame

@dataclass
class ClusterEntity:
    id: int
    title: str
    description: str
    vacancies_count: int
    avg_salary: str
    skills: List[str]
    remote_rate: float

@dataclass
class ClusteringResult:
    method_name: str
    n_clusters: int
    silhouette_score: float
    clusters: List[ClusterEntity]