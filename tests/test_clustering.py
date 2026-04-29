"""
Тесты модулей кластеризации: ClusteringService и ClusteringCache.
Покрывают: выбор алгоритма, оптимизацию k, построение ClusterEntity,
вычисление меток зарплаты, навыки, удалённость, кэш-операции.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.models import ClusterEntity, ClusteringResult
from src.infrastructure.cache import ClusteringCache
from src.services.clustering_service import ClusteringService


# ─────────────────────────────── Фикстуры ────────────────────────────────────

@pytest.fixture
def numeric_only_df():
    """Датафрейм только с числовыми признаками — ожидается K-Means."""
    np.random.seed(0)
    n = 90
    return pd.DataFrame({
        "salary_avg": np.concatenate([
            np.random.normal(60000, 5000, 30),
            np.random.normal(120000, 5000, 30),
            np.random.normal(200000, 5000, 30),
        ]),
        "min_experience_years": np.concatenate([
            np.zeros(30), np.ones(30) * 2, np.ones(30) * 5
        ]),
        "name": ["Аналитик"] * n,
        "schedule_name": ["Полный день"] * n,
        "area_name": ["Москва"] * n,
        "skills_list": [["Python", "SQL"]] * n,
    })


@pytest.fixture
def categorical_only_df():
    """Датафрейм только с категориальными признаками — ожидается K-Modes."""
    np.random.seed(1)
    n = 90
    return pd.DataFrame({
        "salary_avg": [np.nan] * n,
        "min_experience_years": [np.nan] * n,
        "name": np.random.choice(["Аналитик данных", "BI-аналитик", "DA"], n),
        "schedule_name": np.random.choice(["Полный день", "Удаленная работа"], n),
        "area_name": np.random.choice(["Москва", "Казань"], n),
        "skills_list": [["Python"]] * n,
    })


@pytest.fixture
def mixed_df():
    """Датафрейм со смешанными признаками — ожидается K-Prototypes."""
    np.random.seed(2)
    n = 90
    return pd.DataFrame({
        "salary_avg": np.concatenate([
            np.random.normal(70000, 5000, 45),
            np.random.normal(160000, 5000, 45),
        ]),
        "min_experience_years": np.concatenate([np.zeros(45), np.ones(45) * 4]),
        "name": np.random.choice(["Аналитик данных", "BI-аналитик", "DA"], n),
        "schedule_name": np.random.choice(["Полный день", "Удаленная работа"], n),
        "area_name": np.random.choice(["Москва", "Санкт-Петербург"], n),
        "skills_list": [["Python", "SQL"]] * n,
    })


@pytest.fixture
def title_df():
    """Датафрейм, где один title доминирует (>= 30%) — для теста наименования кластера."""
    n = 60
    return pd.DataFrame({
        "salary_avg": np.full(n, 100000.0),
        "min_experience_years": np.zeros(n),
        "name": ["Аналитик данных"] * 50 + ["BI-аналитик"] * 10,
        "schedule_name": ["Полный день"] * n,
        "area_name": ["Москва"] * n,
        "skills_list": [["Python", "SQL", "Pandas", "Excel", "Tableau"]] * n,
    })


@pytest.fixture
def salary_tag_df():
    """Датафрейм с тремя зарплатными уровнями.

    40 низких (30k) + 10 средних (100k) + 10 высоких (500k).
    q30 ≈ 30000, q70 ≈ 100000 — высокий кластер (500k) строго выше q70 → 🟢.
    """
    return pd.DataFrame({
        "salary_avg": [30000.0] * 40 + [100000.0] * 10 + [500000.0] * 10,
        "min_experience_years": [0] * 40 + [2] * 10 + [5] * 10,
        "name": ["Junior"] * 40 + ["Middle"] * 10 + ["Senior"] * 10,
        "schedule_name": ["Полный день"] * 60,
        "area_name": ["Москва"] * 60,
        "skills_list": [["SQL"]] * 60,
    })


@pytest.fixture
def remote_df():
    """Датафрейм с известной долей удалённой работы."""
    n = 60
    schedules = ["Удаленная работа"] * 30 + ["Полный день"] * 30
    return pd.DataFrame({
        "salary_avg": np.concatenate([
            np.random.normal(90000, 5000, 30),
            np.random.normal(90000, 5000, 30),
        ]),
        "min_experience_years": np.zeros(n),
        "name": ["Аналитик данных"] * n,
        "schedule_name": schedules,
        "area_name": ["Москва"] * n,
        "skills_list": [["Python"]] * n,
    })


@pytest.fixture
def skills_df():
    """Датафрейм с контролируемым набором навыков."""
    n = 60
    return pd.DataFrame({
        "salary_avg": np.full(n, 100000.0),
        "min_experience_years": np.zeros(n),
        "name": ["Аналитик данных"] * n,
        "schedule_name": ["Полный день"] * n,
        "area_name": ["Москва"] * n,
        "skills_list": [["Python", "SQL", "Pandas", "Excel", "Tableau", "PowerBI"]] * n,
    })


@pytest.fixture
def empty_cache(tmp_path):
    """ClusteringCache в пустой временной директории."""
    return ClusteringCache(cache_dir=tmp_path)


@pytest.fixture
def sample_clustering_result():
    """Минимальный ClusteringResult для тестов кэша."""
    entity = ClusterEntity(
        id=1, title="Группа #1", description="тест",
        vacancies_count=10, avg_salary="100 000 ₽", skills=["Python"],
        remote_rate=0.0, median_salary="100 000 ₽",
    )
    return ClusteringResult(
        method_name="K-Means",
        n_clusters=1,
        silhouette_score=0.5,
        clusters=[entity],
        k_scores=[(2, 0.5)],
    )


# ─────────────────────────── ClusteringService ───────────────────────────────

class TestAlgorithmSelection:
    """Проверяем, что выбирается правильный алгоритм по типу признаков."""

    def test_numeric_only_uses_kmeans(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(
            ["Зарплата", "Минимальный требуемый опыт"], range(2, 5)
        )
        assert result.method_name == "K-Means"

    def test_categorical_only_uses_kmodes(self, categorical_only_df):
        svc = ClusteringService(categorical_only_df)
        result = svc.perform_clustering(
            ["Название профессии", "График работы", "Регион РФ"], range(2, 4)
        )
        assert result.method_name == "K-Modes"

    def test_mixed_uses_kprototypes(self, mixed_df):
        svc = ClusteringService(mixed_df)
        result = svc.perform_clustering(
            ["Зарплата", "Название профессии"], range(2, 4)
        )
        assert result.method_name == "K-Prototypes"


class TestInputValidation:
    """Проверяем обработку некорректных входных данных."""

    def test_empty_features_raises_valueerror(self, clustering_df):
        svc = ClusteringService(clustering_df)
        with pytest.raises(ValueError, match="Не выбраны признаки"):
            svc.perform_clustering([], range(2, 5))

    def test_all_nan_column_raises_valueerror(self):
        df = pd.DataFrame({
            "salary_avg": [np.nan] * 20,
            "min_experience_years": [np.nan] * 20,
            "name": ["X"] * 20,
            "schedule_name": ["Y"] * 20,
            "area_name": ["Z"] * 20,
            "skills_list": [[]] * 20,
        })
        svc = ClusteringService(df)
        with pytest.raises(ValueError, match="Нет данных"):
            svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 4))


class TestClusteringResult:
    """Проверяем структуру и корректность возвращаемого ClusteringResult."""

    def test_returns_clustering_result_type(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата"], range(2, 4))
        assert isinstance(result, ClusteringResult)

    def test_n_clusters_within_requested_range(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        k_range = range(2, 5)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], k_range)
        assert k_range.start <= result.n_clusters <= k_range.stop

    def test_silhouette_score_is_non_negative(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 5))
        assert 0.0 <= result.silhouette_score <= 1.0

    def test_cluster_entities_count_matches_n_clusters(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 5))
        assert len(result.clusters) == result.n_clusters

    def test_k_scores_populated(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 5))
        assert len(result.k_scores) > 0

    def test_k_scores_contain_tuples_in_range(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        k_range = range(2, 5)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], k_range)
        for k, score in result.k_scores:
            assert k in k_range
            assert 0.0 <= score <= 1.0


class TestClusterEntity:
    """Проверяем поля ClusterEntity."""

    def _get_first_entity(self, df, features, k_range=range(2, 4)):
        svc = ClusteringService(df)
        result = svc.perform_clustering(features, k_range)
        return result.clusters[0]

    def test_entity_has_required_fields(self, numeric_only_df):
        entity = self._get_first_entity(numeric_only_df, ["Зарплата"])
        assert entity.id >= 1
        assert isinstance(entity.title, str) and entity.title
        assert isinstance(entity.vacancies_count, int) and entity.vacancies_count > 0
        assert isinstance(entity.avg_salary, str)
        assert isinstance(entity.median_salary, str)
        assert isinstance(entity.skills, list)
        assert isinstance(entity.remote_rate, float)
        assert 0.0 <= entity.remote_rate <= 100.0
        assert 0.0 <= entity.salary_rate <= 100.0

    def test_entity_id_sequential(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 4))
        ids = [e.id for e in result.clusters]
        assert ids == list(range(1, result.n_clusters + 1))

    def test_vacancies_count_sum_equals_total(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 4))
        total = sum(e.vacancies_count for e in result.clusters)
        # Общий счёт = строки без NaN в выбранных колонках
        expected = numeric_only_df[["salary_avg", "min_experience_years"]].dropna().shape[0]
        assert total == expected

    def test_entity_popular_regions_list(self, clustering_df):
        svc = ClusteringService(clustering_df)
        result = svc.perform_clustering(["Зарплата", "Регион РФ"], range(2, 4))
        for entity in result.clusters:
            assert isinstance(entity.popular_regions, list)


class TestClusterTitle:
    """Проверяем логику автоматического наименования кластеров."""

    def test_dominant_profession_appears_in_title(self, title_df):
        svc = ClusteringService(title_df)
        # Используем только числовые признаки, чтобы кластер включал все строки
        result = svc.perform_clustering(
            ["Зарплата", "Минимальный требуемый опыт"], range(2, 3)
        )
        # Во всём наборе «Аналитик данных» занимает 83%, должен войти в title
        all_titles = " ".join(e.title for e in result.clusters)
        assert "Аналитик данных" in all_titles

    def test_no_dominant_profession_gives_group_name(self):
        """Если ни одна профессия не составляет >= 30%, кластер называется «Группа #N»."""
        np.random.seed(7)
        # 12 профессий × 5 строк = 60; при k=2 каждая = 5/30 ≈ 17% < 30%
        professions = [f"Профессия_{i}" for i in range(12)]
        name_col = [professions[i // 5] for i in range(60)]
        df = pd.DataFrame({
            "salary_avg": np.concatenate([
                np.random.normal(70000, 3000, 30),
                np.random.normal(150000, 3000, 30),
            ]),
            "min_experience_years": [0] * 30 + [5] * 30,
            "name": name_col,
            "schedule_name": ["Полный день"] * 60,
            "area_name": ["Москва"] * 60,
            "skills_list": [[]] * 60,
        })
        svc = ClusteringService(df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 3))
        # Хотя бы один кластер должен иметь «Группа #»
        group_titles = [e for e in result.clusters if "Группа #" in e.title]
        assert len(group_titles) > 0


class TestSalaryTag:
    """Проверяем теги уровня зарплаты 🟢/🟡/🔴."""

    def test_high_salary_cluster_has_green_tag(self, salary_tag_df):
        svc = ClusteringService(salary_tag_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 3))
        tags = [e.title for e in result.clusters]
        assert any("🟢" in t for t in tags), f"Ожидался тег 🟢, получено: {tags}"

    def test_low_salary_cluster_has_red_tag(self, salary_tag_df):
        svc = ClusteringService(salary_tag_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 3))
        tags = [e.title for e in result.clusters]
        assert any("🔴" in t for t in tags), f"Ожидался тег 🔴, получено: {tags}"

    def test_no_salary_data_means_no_tag(self):
        """При отсутствии зарплатных данных тег не добавляется."""
        n = 30
        df = pd.DataFrame({
            "salary_avg": [np.nan] * n,
            "min_experience_years": np.zeros(n),
            "name": np.random.choice(["A", "B", "C"], n),
            "schedule_name": ["Полный день"] * n,
            "area_name": ["Москва"] * n,
            "skills_list": [[]] * n,
        })
        svc = ClusteringService(df)
        result = svc.perform_clustering(
            ["Название профессии", "График работы"], range(2, 3)
        )
        for entity in result.clusters:
            assert "🟢" not in entity.title
            assert "🔴" not in entity.title
            assert "🟡" not in entity.title


class TestTopSkills:
    """Проверяем извлечение топ-5 навыков из skills_list."""

    def test_top_skills_extracted(self, skills_df):
        svc = ClusteringService(skills_df)
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        all_skills = []
        for entity in result.clusters:
            all_skills.extend(entity.skills)
        # Навыки из нашего фикстура должны присутствовать
        assert any(s in all_skills for s in ["Python", "SQL", "Pandas"])

    def test_top_skills_max_five(self, skills_df):
        svc = ClusteringService(skills_df)
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        for entity in result.clusters:
            assert len(entity.skills) <= 5

    def test_skills_list_as_string_parsed(self):
        """skills_list в виде строки '['Python', 'SQL']' должен быть разобран."""
        n = 30
        df = pd.DataFrame({
            "salary_avg": np.full(n, 100000.0),
            "min_experience_years": np.zeros(n),
            "name": ["Аналитик"] * n,
            "schedule_name": ["Полный день"] * n,
            "area_name": ["Москва"] * n,
            "skills_list": ["['Python', 'SQL']"] * n,
        })
        svc = ClusteringService(df)
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        all_skills = [s for e in result.clusters for s in e.skills]
        assert "Python" in all_skills or "SQL" in all_skills

    def test_empty_skills_list_no_error(self):
        n = 30
        df = pd.DataFrame({
            "salary_avg": np.full(n, 100000.0),
            "min_experience_years": np.zeros(n),
            "name": ["Аналитик"] * n,
            "schedule_name": ["Полный день"] * n,
            "area_name": ["Москва"] * n,
            "skills_list": [[]] * n,
        })
        svc = ClusteringService(df)
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        for entity in result.clusters:
            assert entity.skills == []


class TestRemoteRate:
    """Проверяем расчёт доли удалённой работы."""

    def test_remote_rate_fifty_percent(self, remote_df):
        svc = ClusteringService(remote_df)
        # Группируем только по зарплате (числовой) → K-Means
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        all_remote = sum(e.remote_rate * e.vacancies_count for e in result.clusters)
        total = sum(e.vacancies_count for e in result.clusters)
        # Суммарная доля удалённых должна быть близка к 50%
        overall_rate = all_remote / total
        assert 40.0 < overall_rate < 60.0

    def test_remote_rate_zero_when_no_remote(self):
        n = 30
        df = pd.DataFrame({
            "salary_avg": np.full(n, 100000.0),
            "min_experience_years": np.zeros(n),
            "name": ["Аналитик"] * n,
            "schedule_name": ["Полный день"] * n,
            "area_name": ["Москва"] * n,
            "skills_list": [[]] * n,
        })
        svc = ClusteringService(df)
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        for entity in result.clusters:
            assert entity.remote_rate == 0.0

    def test_remote_rate_hundred_when_all_remote(self):
        # Добавляем разброс в зарплате, чтобы K-Means не порождал пустые кластеры.
        np.random.seed(9)
        n = 60
        df = pd.DataFrame({
            "salary_avg": np.concatenate([
                np.random.normal(80000, 5000, 30),
                np.random.normal(160000, 5000, 30),
            ]),
            "min_experience_years": np.zeros(n),
            "name": ["Аналитик"] * n,
            "schedule_name": ["Удаленная работа"] * n,
            "area_name": ["Москва"] * n,
            "skills_list": [[]] * n,
        })
        svc = ClusteringService(df)
        result = svc.perform_clustering(["Зарплата"], range(2, 3))
        # Проверяем только непустые кластеры
        for entity in result.clusters:
            if entity.vacancies_count > 0:
                assert entity.remote_rate == 100.0


class TestFindBestK:
    """Проверяем внутреннюю логику _find_best_k."""

    def test_best_k_has_highest_silhouette(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата", "Минимальный требуемый опыт"], range(2, 6))
        if result.k_scores:
            best_score = max(score for _, score in result.k_scores)
            assert abs(result.silhouette_score - best_score) < 1e-9

    def test_minimum_k_is_two(self, numeric_only_df):
        svc = ClusteringService(numeric_only_df)
        result = svc.perform_clustering(["Зарплата"], range(2, 5))
        assert result.n_clusters >= 2


# ─────────────────────────── ClusteringCache ─────────────────────────────────

class TestClusteringCache:
    """Проверяем операции ClusteringCache: get/put, ключи, обработку ошибок."""

    def test_get_returns_none_on_miss(self, empty_cache):
        result = empty_cache.get("checksum_abc", ["Зарплата"], range(2, 5))
        assert result is None

    def test_put_and_get_round_trip(self, empty_cache, sample_clustering_result):
        empty_cache.put("cs1", ["Зарплата"], range(2, 5), sample_clustering_result)
        retrieved = empty_cache.get("cs1", ["Зарплата"], range(2, 5))
        assert retrieved is not None
        assert retrieved.method_name == sample_clustering_result.method_name
        assert retrieved.n_clusters == sample_clustering_result.n_clusters
        assert retrieved.silhouette_score == sample_clustering_result.silhouette_score

    def test_different_features_different_cache_entry(self, empty_cache, sample_clustering_result):
        empty_cache.put("cs1", ["Зарплата"], range(2, 5), sample_clustering_result)
        # Другие признаки → промах
        result = empty_cache.get("cs1", ["Регион РФ"], range(2, 5))
        assert result is None

    def test_different_checksum_different_cache_entry(self, empty_cache, sample_clustering_result):
        empty_cache.put("cs1", ["Зарплата"], range(2, 5), sample_clustering_result)
        result = empty_cache.get("cs_other", ["Зарплата"], range(2, 5))
        assert result is None

    def test_different_k_range_different_cache_entry(self, empty_cache, sample_clustering_result):
        empty_cache.put("cs1", ["Зарплата"], range(2, 5), sample_clustering_result)
        result = empty_cache.get("cs1", ["Зарплата"], range(2, 8))
        assert result is None

    def test_corrupted_cache_returns_none(self, empty_cache):
        checksum = "corrupt_test"
        features = ["Зарплата"]
        k_range = range(2, 5)
        raw_key = f"v2|{checksum}|{','.join(sorted(features))}|{k_range.start}-{k_range.stop}"
        key = hashlib.md5(raw_key.encode()).hexdigest()
        cache_file = empty_cache.cache_dir / f"clustering_{key}.pkl"
        cache_file.write_bytes(b"not_a_valid_joblib_file")
        result = empty_cache.get(checksum, features, k_range)
        assert result is None

    def test_corrupted_cache_file_deleted(self, empty_cache):
        checksum = "corrupt_delete"
        features = ["Зарплата"]
        k_range = range(2, 5)
        raw_key = f"v2|{checksum}|{','.join(sorted(features))}|{k_range.start}-{k_range.stop}"
        key = hashlib.md5(raw_key.encode()).hexdigest()
        cache_file = empty_cache.cache_dir / f"clustering_{key}.pkl"
        cache_file.write_bytes(b"garbage")
        empty_cache.get(checksum, features, k_range)
        assert not cache_file.exists()

    def test_cache_dir_created_on_init(self, tmp_path):
        new_dir = tmp_path / "new_subdir" / "cache"
        assert not new_dir.exists()
        ClusteringCache(cache_dir=new_dir)
        assert new_dir.exists()

    def test_wrong_type_in_cache_returns_none(self, empty_cache, tmp_path):
        import joblib
        checksum = "wrong_type"
        features = ["Зарплата"]
        k_range = range(2, 5)
        raw_key = f"v2|{checksum}|{','.join(sorted(features))}|{k_range.start}-{k_range.stop}"
        key = hashlib.md5(raw_key.encode()).hexdigest()
        cache_file = empty_cache.cache_dir / f"clustering_{key}.pkl"
        joblib.dump({"not": "a ClusteringResult"}, cache_file)
        result = empty_cache.get(checksum, features, k_range)
        assert result is None

    def test_make_key_is_md5_hex(self, empty_cache):
        key = empty_cache._make_key("cs", ["Зарплата"], range(2, 5))
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_make_key_features_order_independent(self, empty_cache):
        k1 = empty_cache._make_key("cs", ["Зарплата", "Регион РФ"], range(2, 5))
        k2 = empty_cache._make_key("cs", ["Регион РФ", "Зарплата"], range(2, 5))
        assert k1 == k2

    def test_multiple_puts_same_key_overwrites(self, empty_cache, sample_clustering_result):
        empty_cache.put("cs1", ["Зарплата"], range(2, 5), sample_clustering_result)
        modified = ClusteringResult(
            method_name="K-Modes",
            n_clusters=3,
            silhouette_score=0.9,
            clusters=sample_clustering_result.clusters,
            k_scores=[],
        )
        empty_cache.put("cs1", ["Зарплата"], range(2, 5), modified)
        retrieved = empty_cache.get("cs1", ["Зарплата"], range(2, 5))
        assert retrieved.method_name == "K-Modes"
        assert retrieved.n_clusters == 3
