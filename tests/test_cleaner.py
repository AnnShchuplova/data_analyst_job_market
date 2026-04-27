"""
Тесты для src.data_processing.cleaner.DataCleaner.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest


class TestCalculateAvgSalary:
    """Тесты метода _calculate_avg_salary."""

    def test_both_values(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        assert cleaner._calculate_avg_salary(100000, 150000) == 125000.0

    def test_only_from(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        assert cleaner._calculate_avg_salary(100000, np.nan) == 100000.0

    def test_only_to(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        assert cleaner._calculate_avg_salary(np.nan, 200000) == 200000.0

    def test_both_none(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner._calculate_avg_salary(np.nan, np.nan)
        assert np.isnan(result)

    def test_both_zero(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        assert cleaner._calculate_avg_salary(0, 0) == 0.0


class TestConvertStringDicts:
    """Тесты метода _convert_string_dicts."""

    def test_dict_string(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": ['{"key": "value"}']})
        result = cleaner._convert_string_dicts(df)

        assert isinstance(result["col"].iloc[0], dict)
        assert result["col"].iloc[0]["key"] == "value"

    def test_list_string(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": ['["a", "b", "c"]']})
        result = cleaner._convert_string_dicts(df)

        assert isinstance(result["col"].iloc[0], list)
        assert result["col"].iloc[0] == ["a", "b", "c"]

    def test_already_dict(self):
        """Если значение уже dict — не должно меняться."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": [{"key": "value"}]})
        result = cleaner._convert_string_dicts(df)

        assert result["col"].iloc[0] == {"key": "value"}

    def test_already_list(self):
        """Если значение уже list — не должно меняться."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": [[1, 2, 3]]})
        result = cleaner._convert_string_dicts(df)

        assert result["col"].iloc[0] == [1, 2, 3]

    def test_none_value(self):
        """None должен оставаться None."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": [None]})
        result = cleaner._convert_string_dicts(df)

        assert pd.isna(result["col"].iloc[0])

    def test_empty_dict_string(self):
        """Пустой dict как строка должен распарситься."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": ["{}"]})
        result = cleaner._convert_string_dicts(df)

        assert isinstance(result["col"].iloc[0], dict)
        assert len(result["col"].iloc[0]) == 0

    def test_empty_list_string(self):
        """Пустой list как строка должен распарситься."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": ["[]"]})
        result = cleaner._convert_string_dicts(df)

        assert isinstance(result["col"].iloc[0], list)
        assert len(result["col"].iloc[0]) == 0

    def test_plain_string_unchanged(self):
        """Обычная строка без { или [ не должна меняться."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": ["just a string"]})
        result = cleaner._convert_string_dicts(df)

        assert result["col"].iloc[0] == "just a string"

    def test_numeric_column_unchanged(self):
        """Числовая колонка не должна изменяться."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = cleaner._convert_string_dicts(df)

        assert list(result["col"]) == [1, 2, 3]


class TestCleanSalary:
    """Тесты метода clean_salary."""

    def test_extracts_salary_fields_from_dict(self, sample_salary_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_salary(sample_salary_data)

        assert "salary_from" in result.columns
        assert "salary_to" in result.columns
        assert "salary_currency" in result.columns
        assert "has_salary" in result.columns
        assert "salary_avg" in result.columns

    def test_salary_from_extraction(self, sample_salary_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_salary(sample_salary_data)

        assert result["salary_from"].iloc[0] == 80000
        assert result["salary_to"].iloc[0] == 120000
        assert result["salary_currency"].iloc[0] == "RUR"

    def test_non_dict_salary_becomes_none(self, sample_salary_data):
        """Строка 'not_a_dict' в salary должна дать None для from/to."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_salary(sample_salary_data)

        assert pd.isna(result["salary_from"].iloc[4])
        assert pd.isna(result["salary_to"].iloc[4])

    def test_has_salary_flag(self, sample_salary_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_salary(sample_salary_data)

        # Первые три имеют хотя бы одну границу
        assert result["has_salary"].iloc[0] == True
        assert result["has_salary"].iloc[1] == True
        assert result["has_salary"].iloc[2] == True
        # Четвёртая — нет
        assert result["has_salary"].iloc[3] == False

    def test_salary_avg_calculation(self, sample_salary_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_salary(sample_salary_data)

        # (80000 + 120000) / 2 = 100000
        assert result["salary_avg"].iloc[0] == 100000.0

    def test_no_salary_column_raises(self):
        """Если колонки salary нет — метод выбрасывает KeyError (известный баг кода)."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"name": ["V1"]})

        with pytest.raises(KeyError):
            cleaner.clean_salary(df)

    def test_returns_copy(self, sample_salary_data):
        """Метод не должен мутировать исходный DataFrame."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        original_cols = set(sample_salary_data.columns)
        cleaner.clean_salary(sample_salary_data)

        assert set(sample_salary_data.columns) == original_cols


class TestCleanExperience:
    """Тесты метода clean_experience."""

    def test_extracts_experience_fields(self, sample_experience_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_experience(sample_experience_data)

        assert "experience_id" in result.columns
        assert "experience_name" in result.columns
        assert "min_experience_years" in result.columns
        assert "avg_experience_years" in result.columns

    def test_experience_id_mapping(self, sample_experience_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_experience(sample_experience_data)

        assert result["experience_id"].iloc[0] == "noExperience"
        assert result["experience_name"].iloc[0] == "Нет опыта"

    def test_min_experience_years(self, sample_experience_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_experience(sample_experience_data)

        assert result["min_experience_years"].iloc[0] == 0
        assert result["min_experience_years"].iloc[1] == 1
        assert result["min_experience_years"].iloc[2] == 3
        assert result["min_experience_years"].iloc[3] == 6

    def test_avg_experience_years(self, sample_experience_data):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_experience(sample_experience_data)

        assert result["avg_experience_years"].iloc[0] == 0
        assert result["avg_experience_years"].iloc[1] == 2
        assert result["avg_experience_years"].iloc[2] == 4
        assert result["avg_experience_years"].iloc[3] == 8

    def test_none_experience(self, sample_experience_data):
        """None в experience не должен вызывать ошибку."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        result = cleaner.clean_experience(sample_experience_data)

        # Последняя строка: experience = None
        assert pd.isna(result["experience_id"].iloc[4])
        assert pd.isna(result["min_experience_years"].iloc[4])

    def test_no_experience_column(self):
        """Если колонки experience нет — метод не должен падать."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"name": ["V1"]})
        result = cleaner.clean_experience(df)

        assert "name" in result.columns


class TestCleanEmployer:
    """Тесты метода clean_employer."""

    def test_extracts_employer_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "employer": [
                {"id": "1", "name": "Яндекс"},
                {"id": "2", "name": "Сбер"},
                None,
            ]
        })
        result = cleaner.clean_employer(df)

        assert "employer_name" in result.columns
        assert "employer_id" in result.columns
        assert result["employer_name"].iloc[0] == "Яндекс"
        assert result["employer_name"].iloc[1] == "Сбер"
        assert pd.isna(result["employer_name"].iloc[2])


class TestCleanArea:
    """Тесты метода clean_area."""

    def test_extracts_area_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "area": [
                {"id": "1", "name": "Москва"},
                {"id": "2", "name": "Казань"},
            ]
        })
        result = cleaner.clean_area(df)

        assert "area_name" in result.columns
        assert "area_id" in result.columns
        assert result["area_name"].iloc[0] == "Москва"


class TestCleanSchedule:
    """Тесты метода clean_schedule_field."""

    def test_extracts_schedule_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "schedule": [
                {"id": "fullDay", "name": "Полный день"},
                {"id": "remote", "name": "Удаленная работа"},
            ]
        })
        result = cleaner.clean_schedule_field(df)

        assert "schedule_name" in result.columns
        assert "schedule_id" in result.columns
        assert result["schedule_name"].iloc[0] == "Полный день"


class TestCleanEmployment:
    """Тесты метода clean_employment_field."""

    def test_extracts_employment_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "employment": [
                {"id": "full", "name": "Полная занятость"},
            ]
        })
        result = cleaner.clean_employment_field(df)

        assert "employment_name" in result.columns
        assert result["employment_name"].iloc[0] == "Полная занятость"


class TestCleanProfessionalRoles:
    """Тесты метода clean_professional_roles."""

    def test_extracts_main_role(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "professional_roles": [
                [{"id": "156", "name": "Аналитик данных"}],
                [{"id": "156", "name": "Аналитик данных"}, {"id": "10", "name": "Другая"}],
                [],
                None,
            ]
        })
        result = cleaner.clean_professional_roles(df)

        assert result["main_role_name"].iloc[0] == "Аналитик данных"
        assert result["main_role_name"].iloc[1] == "Аналитик данных"
        assert pd.isna(result["main_role_name"].iloc[2])
        assert pd.isna(result["main_role_name"].iloc[3])


class TestCleanAddressField:
    """Тесты метода clean_address_field."""

    def test_extracts_address_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "address": [
                {"city": "Москва", "street": "Тверская", "building": "1", "raw": "Москва, Тверская 1"},
                None,
            ]
        })
        result = cleaner.clean_address_field(df)

        assert "address_city" in result.columns
        assert "address_street" in result.columns
        assert "address_building" in result.columns
        assert "address_raw" in result.columns
        assert result["address_city"].iloc[0] == "Москва"
        assert pd.isna(result["address_city"].iloc[1])


class TestCleanSnippetField:
    """Тесты метода clean_snippet_field."""

    def test_extracts_requirement_and_responsibility(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "snippet": [
                {"requirement": "Знание Python", "responsibility": "Анализ данных"},
                "not_a_dict",
                None,
            ]
        })
        result = cleaner.clean_snippet_field(df)

        assert "requirement" in result.columns
        assert "responsibility" in result.columns
        assert result["requirement"].iloc[0] == "Знание Python"
        assert result["responsibility"].iloc[0] == "Анализ данных"
        assert pd.isna(result["requirement"].iloc[1])
        assert pd.isna(result["requirement"].iloc[2])


class TestCleanWorkFormat:
    """Тесты метода clean_work_format_field."""

    def test_extracts_work_format_from_list(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "work_format": [
                [{"id": "remote", "name": "Удаленная работа"}],
                [],
                None,
            ]
        })
        result = cleaner.clean_work_format_field(df)

        assert "work_format_name" in result.columns
        assert result["work_format_name"].iloc[0] == "Удаленная работа"
        assert pd.isna(result["work_format_name"].iloc[1])
        assert pd.isna(result["work_format_name"].iloc[2])


class TestCleanDates:
    """Тесты метода clean_dates."""

    def test_parses_dates(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "published_at": ["2025-12-01T10:00:00+0300", "2025-11-15T09:00:00+0300"],
        })
        result = cleaner.clean_dates(df)

        assert "published_date" in result.columns
        assert "published_year_month" in result.columns
        assert "days_since_publication" in result.columns
        assert result["published_year_month"].iloc[0] == "2025-12"

    def test_no_published_at_column(self):
        """Если колонки нет — метод не должен падать."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"name": ["V1"]})
        result = cleaner.clean_dates(df)

        assert "name" in result.columns


class TestExtractSkills:
    """Тесты методов extract_skills / extract_skills_from_text."""

    def test_extract_skills_creates_columns(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "requirement": ["Знание Python и SQL, опыт с PostgreSQL"],
            "responsibility": ["Построение дашбордов в Power BI"],
        })
        result = cleaner.extract_skills(df)

        assert "skills_list" in result.columns
        assert "skills_count" in result.columns
        assert isinstance(result["skills_list"].iloc[0], list)
        assert result["skills_count"].iloc[0] > 0

    def test_detects_python_skill(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "requirement": ["Требуется знание Python"],
        })
        result = cleaner.extract_skills(df)

        assert "Python" in result["skills_list"].iloc[0]

    def test_detects_sql_skill(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "requirement": ["Уверенное владение SQL и PostgreSQL"],
        })
        result = cleaner.extract_skills(df)

        assert "SQL" in result["skills_list"].iloc[0]
        assert "PostgreSQL" in result["skills_list"].iloc[0]

    def test_detects_power_bi(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "responsibility": ["Создание отчётов в Power BI"],
        })
        result = cleaner.extract_skills(df)

        assert "Power BI" in result["skills_list"].iloc[0]

    def test_no_skills_in_text(self):
        """Текст без ключевых слов навыков должен дать пустой список."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "requirement": ["Обязанности описаны общими словами без специфики"],
        })
        result = cleaner.extract_skills(df)

        assert result["skills_list"].iloc[0] == []
        assert result["skills_count"].iloc[0] == 0

    def test_nan_requirement_no_crash(self):
        """NaN в requirement/responsibility не должен вызывать ошибку."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "requirement": [None],
            "responsibility": [np.nan],
        })
        result = cleaner.extract_skills(df)

        assert result["skills_list"].iloc[0] == []

    def test_skills_from_requirement_and_responsibility_merged(self):
        """Навыки из requirement и responsibility должны объединяться."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "requirement": ["Python"],
            "responsibility": ["Power BI"],
        })
        result = cleaner.extract_skills(df)

        skills = result["skills_list"].iloc[0]
        assert "Python" in skills
        assert "Power BI" in skills


class TestRemoveOutliers:
    """Тесты метода remove_outliers_optional."""

    def test_removes_outliers_when_enabled(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        np.random.seed(42)
        normal_salaries = np.random.normal(100000, 15000, 50)
        salaries = list(normal_salaries) + [5000000]
        df = pd.DataFrame({"salary_avg": salaries})
        result = cleaner.remove_outliers_optional(df, remove_outliers=True)

        assert len(result) < 51
        assert 5000000 not in result["salary_avg"].values

    def test_no_removal_when_disabled(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "salary_avg": [50000, 60000, 70000, 80000, 90000, 100000, 5000000],
        })
        result = cleaner.remove_outliers_optional(df, remove_outliers=False)

        assert len(result) == 7

    def test_no_salary_avg_column(self):
        """Если колонки salary_avg нет — метод не должен падать."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"name": ["V1", "V2"]})
        result = cleaner.remove_outliers_optional(df, remove_outliers=True)

        assert len(result) == 2


class TestRemoveJsonFields:
    """Тесты метода remove_json_fields."""

    def test_removes_original_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "salary": [{"from": 100, "to": 200}],
            "salary_from": [100],
            "salary_to": [200],
            "experience": [{"id": "1"}],
            "experience_id": ["1"],
        })
        result = cleaner.remove_json_fields(df)

        assert "salary" not in result.columns
        assert "experience" not in result.columns
        # Извлечённые поля остаются
        assert "salary_from" in result.columns
        assert "experience_id" in result.columns

    def test_removes_service_fields(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "url": ["https://hh.ru/vacancy/1"],
            "alternate_url": ["https://hh.ru/vacancy/1"],
            "name": ["Вакансия"],
        })
        result = cleaner.remove_json_fields(df)

        assert "url" not in result.columns
        assert "alternate_url" not in result.columns
        assert "name" in result.columns


class TestCleanTypeAndDepartment:
    """Тесты clean_type_field и clean_department_field."""

    def test_clean_type(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "type": [{"id": "open", "name": "Открытая"}],
        })
        result = cleaner.clean_type_field(df)

        assert result["type_name"].iloc[0] == "Открытая"
        assert result["type_id"].iloc[0] == "open"

    def test_clean_department(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "department": [{"id": "1", "name": "IT-отдел"}],
        })
        result = cleaner.clean_department_field(df)

        assert result["department_name"].iloc[0] == "IT-отдел"
        assert result["department_id"].iloc[0] == "1"


class TestLoadData:
    """Тесты метода load_data."""

    def test_loads_json_file(self, sample_vacancy_json):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = cleaner.load_data(sample_vacancy_json)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "name" in df.columns

    def test_loads_and_converts_dicts(self, sample_vacancy_json):
        """load_data должен автоматически конвертировать строковые dict/list."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = cleaner.load_data(sample_vacancy_json)

        # salary должен быть dict, а не строкой
        assert isinstance(df["salary"].iloc[0], dict)
        # professional_roles должен быть list
        assert isinstance(df["professional_roles"].iloc[0], list)

    def test_file_not_found(self):
        """При отсутствии файла должен выбрасываться FileNotFoundError."""
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        with pytest.raises(FileNotFoundError):
            cleaner.load_data("/nonexistent/path/file.json")


class TestGetTopSkills:
    """Тесты метода _get_top_skills."""

    def test_returns_top_n(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({
            "skills_list": [
                ["Python", "SQL", "Excel"],
                ["Python", "SQL"],
                ["SQL", "Power BI"],
            ]
        })
        top = cleaner._get_top_skills(df, top_n=2)

        assert len(top) == 2
        assert "SQL" in top  # SQL встречается 3 раза — самый частый
        assert "Python" in top

    def test_empty_skills_list(self):
        from src.data_processing.cleaner import DataCleaner

        cleaner = DataCleaner()
        df = pd.DataFrame({"skills_list": [[] for _ in range(3)]})
        top = cleaner._get_top_skills(df, top_n=5)

        assert top == []
