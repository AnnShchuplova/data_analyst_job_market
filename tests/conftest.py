"""
Общие фикстуры для всех тестов.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Добавляем корень проекта в sys.path, чтобы импорты src.* работали
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ===================== Cleaner fixtures =====================

@pytest.fixture
def sample_salary_data():
    """Датафрейм с колонкой salary (dict) и другими полями."""
    return pd.DataFrame({
        "salary": [
            {"from": 80000, "to": 120000, "currency": "RUR"},
            {"from": 100000, "to": None, "currency": "RUR"},
            {"from": None, "to": 200000, "currency": "RUR"},
            {"from": None, "to": None, "currency": None},
            "not_a_dict",
        ],
        "name": ["Vacancy1", "Vacancy2", "Vacancy3", "Vacancy4", "Vacancy5"],
    })


@pytest.fixture
def sample_experience_data():
    """Датафрейм с колонкой experience (dict)."""
    return pd.DataFrame({
        "experience": [
            {"id": "noExperience", "name": "Нет опыта"},
            {"id": "between1And3", "name": "От 1 года до 3 лет"},
            {"id": "between3And6", "name": "От 3 до 6 лет"},
            {"id": "moreThan6", "name": "Более 6 лет"},
            None,
        ],
        "name": ["V1", "V2", "V3", "V4", "V5"],
    })


@pytest.fixture
def sample_vacancy_json(tmp_path):
    """JSON-файл с сырыми вакансиями для load_data()."""
    import json
    data = [
        {
            "id": "1",
            "name": "Аналитик данных",
            "salary": {"from": 100000, "to": 150000, "currency": "RUR"},
            "experience": {"id": "between1And3", "name": "От 1 года до 3 лет"},
            "employer": {"id": "1", "name": "ООО Компания"},
            "area": {"id": "1", "name": "Москва"},
            "schedule": {"id": "fullDay", "name": "Полный день"},
            "employment": {"id": "full", "name": "Полная занятость"},
            "professional_roles": [{"id": "156", "name": "Аналитик данных"}],
            "address": {"city": "Москва", "street": "Тверская", "building": "1", "raw": "Москва, Тверская 1"},
            "snippet": {"requirement": "Знание Python и SQL", "responsibility": "Анализ данных"},
            "work_format": [{"id": "remote", "name": "Удаленная работа"}],
            "published_at": "2025-12-01T10:00:00+0300",
            "type": {"id": "open", "name": "Открытая"},
        }
    ]
    path = tmp_path / "vacancies.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ===================== Repository fixtures =====================

@pytest.fixture
def sample_csv_data():
    """Простой DataFrame для CSV-репозитория."""
    return pd.DataFrame({
        "name": ["Аналитик", "Разработчик", "Аналитик"],
        "salary_avg": [100000, 150000, 120000],
        "area_name": ["Москва", "Санкт-Петербург", "Казань"],
    })


# ===================== Clustering fixtures =====================

@pytest.fixture
def clustering_df():
    """Датафрейм с числовыми и категориальными признаками для кластеризации."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "salary_avg": np.random.randint(60000, 200000, n),
        "min_experience_years": np.random.choice([0, 1, 3, 6], n),
        "name": np.random.choice(
            ["Аналитик данных", "BI-аналитик", "Системный аналитик", "DA"], n
        ),
        "schedule_name": np.random.choice(
            ["Полный день", "Удаленная работа", "Гибкий график"], n
        ),
        "area_name": np.random.choice(
            ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"], n
        ),
        "skills_list": [
            ["Python", "SQL", "Pandas"],
            ["Excel", "Power BI"],
            ["Python", "Machine Learning"],
            ["SQL"],
        ] * 15,
    })


# ===================== ML fixtures =====================

@pytest.fixture
def salary_regression_data():
    """Синтетический датасет для регрессии зарплат (достаточно большой для ML)."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "experience_name": np.random.choice(
            ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"], n
        ),
        "area_name": np.random.choice(["Москва", "Санкт-Петербург", "Казань", "Новосибирск"], n),
        "main_role_name": np.random.choice(
            ["Аналитик данных", "BI-аналитик", "Системный аналитик"], n
        ),
        "schedule_name": np.random.choice(["Полный день", "Удаленная работа", "Гибкий график"], n),
        "work_format_name": np.random.choice(["Офис", "Удаленная работа", "Гибрид"], n),
        "employment_name": np.random.choice(["Полная", "Частичная", "Проектная"], n),
        "requirement": ["Знание Python и SQL"] * n,
        "responsibility": ["Анализ данных, построение дашбордов"] * n,
        "skill_count": np.random.randint(1, 10, n),
    })

    # Добавляем целевую переменную с реальной зависимостью
    base_salary = 60000
    exp_bonus = df["experience_name"].map({
        "Нет опыта": 0,
        "От 1 года до 3 лет": 30000,
        "От 3 до 6 лет": 60000,
        "Более 6 лет": 90000,
    })
    region_bonus = df["area_name"].map({
        "Москва": 40000,
        "Санкт-Петербург": 25000,
        "Казань": 10000,
        "Новосибирск": 8000,
    })
    noise = np.random.normal(0, 15000, n)
    df["target_salary"] = base_salary + exp_bonus + region_bonus + noise
    df["target_salary"] = df["target_salary"].clip(lower=30000)

    return df


@pytest.fixture
def time_series_daily():
    """Синтетический дневной временной ряд вакансий (90 дней)."""
    np.random.seed(42)
    dates = pd.date_range("2025-10-01", periods=90, freq="D")
    # Добавляем тренд + недельную сезонность + шум
    trend = np.linspace(10, 30, 90)
    seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, 90))
    noise = np.random.normal(0, 2, 90)
    values = np.maximum(0, trend + seasonal + noise).astype(int)
    return pd.Series(values, index=dates, name="vacancies_count")
