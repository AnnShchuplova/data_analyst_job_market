"""
Модуль генерации признаков для ML-моделей предсказания зарплаты.
Создаёт числовые, бинарные, TF-IDF и категориальные признаки.

Целевая колонка: salary_avg
Колонка навыков: skills_list
Текстовые колонки: requirement, responsibility
"""

import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Генератор признаков для предсказания зарплаты."""

    # Категориальные строковые признаки (CatBoost — как есть, RF — ordinal)
    CAT_COLUMNS = [
        'experience_name', 'area_name', 'main_role_name',
        'schedule_name', 'work_format_name', 'employment_name'
    ]

    # Числовые признаки
    NUMERIC_COLUMNS = [
        'estimated_experience_years', 'skills_count',
        'days_since_publication', 'experience_id',
        'min_experience_years', 'avg_experience_years'
    ]

    # Бинарные признаки
    BINARY_COLUMNS = [
        'has_test', 'premium', 'response_letter_required', 'accept_labor_contract'
    ]

    # Текстовые колонки — для CatBoost передаём как text_features,
    # для RF/GB делаем TF-IDF
    TEXT_COLUMNS = ['requirement', 'responsibility']

    # Целевая переменная
    TARGET = 'salary_avg'

    # Ключевые слова уровня должности (из названия вакансии)
    SENIORITY_KEYWORDS = {
        'senior': ['senior', 'старший', 'ведущий', 'lead', 'руководитель',
                    'principal', 'chief', 'главный', 'head', 'director',
                    'начальник', 'куратор'],
        'middle': ['middle', 'уверенный'],
        'junior': ['junior', 'младший', 'стажёр', 'стажер', 'intern',
                    'помощник', 'assistant', 'trainee'],
    }

    def __init__(self, top_n_skills=20, tfidf_max_features=20, text_tfidf_max=30):
        self.top_n_skills = top_n_skills
        self.tfidf_max_features = tfidf_max_features
        self.text_tfidf_max = text_tfidf_max
        self._text_mean_len = 0
        self.tfidf_vectorizer = None
        self.text_vectorizer = None
        self.top_skills = []
        self.skill_columns = []
        self.label_encoders = {}
        self.top_employers = []

    # -------------------------------------------------------
    # Парсинг навыков
    # -------------------------------------------------------
    @staticmethod
    def _parse_skills(skills_raw) -> list:
        """Извлечение списка навыков из строки или списка."""
        if pd.isna(skills_raw):
            return []
        if isinstance(skills_raw, list):
            return [str(s).strip().lower() for s in skills_raw if str(s).strip()]
        if isinstance(skills_raw, str):
            text = skills_raw.strip()
            if text.startswith('[') and text.endswith(']'):
                text = text[1:-1]
            skills = []
            for part in text.split(','):
                s = part.strip().strip("'\"").lower()
                if s:
                    skills.append(s)
            return skills
        return []

    # -------------------------------------------------------
    # TF-IDF навыков
    # -------------------------------------------------------
    def _fit_tfidf(self, df):
        """Обучение TF-IDF векторизатора на навыках."""
        texts = df['skills_list'].apply(
            lambda x: ' '.join(self._parse_skills(x))
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            min_df=3,
            max_df=0.85,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_vectorizer.fit(texts)
        n = len(self.tfidf_vectorizer.get_feature_names_out())
        logger.info(f"TF-IDF навыков: {n} признаков")

    def _transform_tfidf(self, df):
        """Применение обученного TF-IDF навыков."""
        texts = df['skills_list'].apply(
            lambda x: ' '.join(self._parse_skills(x))
        )
        matrix = self.tfidf_vectorizer.transform(texts)
        cols = [f'skill_tfidf_{i}' for i in range(matrix.shape[1])]
        return pd.DataFrame(matrix.toarray(), columns=cols, index=df.index)

    # -------------------------------------------------------
    # TF-IDF текста вакансии (для RF/GB, не для CatBoost)
    # -------------------------------------------------------
    def _fit_text_tfidf(self, df):
        """Обучение TF-IDF на тексте (только для RF/GB)."""
        texts = df[self.TEXT_COLUMNS].fillna('').agg(' '.join, axis=1)
        self.text_vectorizer = TfidfVectorizer(
            max_features=self.text_tfidf_max,
            min_df=5,
            max_df=0.9,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'
        )
        self.text_vectorizer.fit(texts)
        n = len(self.text_vectorizer.get_feature_names_out())
        logger.info(f"TF-IDF текста вакансии (для RF/GB): {n} признаков")

    def _transform_text_tfidf(self, df):
        """Применение TF-IDF текста."""
        texts = df[self.TEXT_COLUMNS].fillna('').agg(' '.join, axis=1)
        matrix = self.text_vectorizer.transform(texts)
        cols = [f'text_tfidf_{i}' for i in range(matrix.shape[1])]
        return pd.DataFrame(matrix.toarray(), columns=cols, index=df.index)

    # -------------------------------------------------------
    # Бинарные признаки топ-навыков
    # -------------------------------------------------------
    def _fit_top_skills(self, df):
        """Определение топ-навыков по частоте."""
        all_skills = []
        for raw in df['skills_list']:
            all_skills.extend(self._parse_skills(raw))
        counts = Counter(all_skills)
        self.top_skills = [s for s, _ in counts.most_common(self.top_n_skills)]
        self.skill_columns = []
        for skill in self.top_skills:
            col = f'skill_has_{re.sub(r"[^a-z0-9а-яё]", "_", skill)}'
            self.skill_columns.append(col)
        logger.info(f"Топ-{len(self.top_skills)} навыков: {self.top_skills[:10]}...")

    def _transform_top_skills(self, df):
        """Бинарные признаки топ-навыков (0/1)."""
        result = {}
        for skill in self.top_skills:
            col = f'skill_has_{re.sub(r"[^a-z0-9а-яё]", "_", skill)}'
            result[col] = df['skills_list'].apply(
                lambda x, s=skill: int(s in self._parse_skills(x))
            )
        return pd.DataFrame(result, index=df.index)

    # -------------------------------------------------------
    # Ordinal encoding (для RandomForest)
    # -------------------------------------------------------
    def _fit_label_encoders(self, df):
        """Обучение LabelEncoder для категориальных колонок."""
        for col in self.CAT_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].fillna('Не указано').astype(str))
                self.label_encoders[col] = le
        logger.info(
            f"Ordinal encoding: {len(self.label_encoders)} категориальных колонок"
        )

    def _transform_label_encoders(self, df):
        """Применение LabelEncoder."""
        result = {}
        for col, le in self.label_encoders.items():
            if col in df.columns:
                encoded = df[col].fillna('Не указано').astype(str)
                classes = set(le.classes_)
                encoded = encoded.apply(
                    lambda x: x if x in classes else 'Не указано'
                )
                result[f'{col}_enc'] = le.transform(encoded)
        return pd.DataFrame(result, index=df.index)

    # -------------------------------------------------------
    # Признаки из названия вакансии (seniority level)
    # -------------------------------------------------------
    def _add_title_features(self, df):
        """Извлечение уровня должности из названия вакансии."""
        result = pd.DataFrame(index=df.index)

        if 'name' not in df.columns:
            logger.warning("Колонка 'name' не найдена, признаки из названия пропущены")
            return result

        titles = df['name'].fillna('').astype(str).str.lower()

        for level, keywords in self.SENIORITY_KEYWORDS.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            result[f'is_{level}'] = titles.str.contains(pattern, na=False).astype(int)

        result['seniority_score'] = (
            result.get('is_senior', 0) * 2 +
            result.get('is_middle', 0) * 1 +
            result.get('is_junior', 0) * (-1)
        )
        no_level = (
            (result.get('is_senior', 0) == 0) &
            (result.get('is_middle', 0) == 0) &
            (result.get('is_junior', 0) == 0)
        )
        result.loc[no_level, 'seniority_score'] = 1

        logger.info(
            f"Признаки из названия: "
            f"senior={result['is_senior'].sum()}, "
            f"middle={result['is_middle'].sum()}, "
            f"junior={result['is_junior'].sum()}"
        )
        return result

    # -------------------------------------------------------
    # Признаки работодателя
    # -------------------------------------------------------
    def _fit_employer_features(self, df):
        """Определение топ-работодателей."""
        if 'employer_name' not in df.columns:
            return
        employer_counts = df['employer_name'].value_counts()
        self.top_employers = set(
            employer_counts[employer_counts >= 5].index
        )
        logger.info(f"Топ-работодатели (5+ вакансий): {len(self.top_employers)} компаний")

    def _add_employer_features(self, df):
        """Признаки на основе работодателя."""
        result = pd.DataFrame(index=df.index)
        if 'employer_name' not in df.columns:
            return result

        result['is_large_employer'] = df['employer_name'].apply(
            lambda x: int(x in self.top_employers) if pd.notna(x) else 0
        )
        result['employer_name_len'] = df['employer_name'].fillna('').str.len()
        return result

    # -------------------------------------------------------
    # Текстовые статистики
    # -------------------------------------------------------
    def _add_text_stats(self, df):
        """Признаки длины текста требований и обязанностей."""
        stats = pd.DataFrame(index=df.index)
        for col in self.TEXT_COLUMNS:
            if col in df.columns:
                stats[f'{col}_len'] = df[col].fillna('').str.len()

        text_cols_present = [c for c in self.TEXT_COLUMNS if c in df.columns]
        if text_cols_present:
            stats['total_text_len'] = df[text_cols_present].fillna('').agg(
                lambda row: sum(len(str(v)) for v in row), axis=1
            )

        self._text_mean_len = stats['total_text_len'].mean() if 'total_text_len' in stats else 0
        return stats

    # -------------------------------------------------------
    # Основные методы
    # -------------------------------------------------------
    def fit_transform(self, df):
        """
        Обучение трансформаций и генерация признаков.
        Возвращает: (features_df, skill_columns)

        В features_df ОСТАВЛЯЮТСЯ текстовые колонки (requirement, responsibility)
        для CatBoost text_features. Для RF/GB они убираются в _prepare_rf_features.
        """
        logger.info(f"Генерация признаков: {len(df)} строк")
        result = pd.DataFrame(index=df.index)

        # --- Числовые ---
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- Бинарные ---
        for col in self.BINARY_COLUMNS:
            if col in df.columns:
                result[col] = df[col].fillna(0).astype(int)

        # --- Категориальные (строковые, для CatBoost) ---
        for col in self.CAT_COLUMNS:
            if col in df.columns:
                result[col] = df[col].fillna('Не указано').astype(str)

        # --- Ordinal encoding ---
        self._fit_label_encoders(df)
        enc_df = self._transform_label_encoders(df)
        result = pd.concat([result, enc_df], axis=1)

        # --- Признаки из названия вакансии ---
        title_df = self._add_title_features(df)
        result = pd.concat([result, title_df], axis=1)

        # --- Признаки работодателя ---
        self._fit_employer_features(df)
        employer_df = self._add_employer_features(df)
        result = pd.concat([result, employer_df], axis=1)

        # --- TF-IDF навыков ---
        if 'skills_list' in df.columns:
            self._fit_tfidf(df)
            tfidf_df = self._transform_tfidf(df)
            result = pd.concat([result, tfidf_df], axis=1)

        # --- Бинарные признаки топ-навыков ---
        if 'skills_list' in df.columns:
            self._fit_top_skills(df)
            skills_df = self._transform_top_skills(df)
            result = pd.concat([result, skills_df], axis=1)

        # --- TF-IDF текста (числовые, для RF/GB) ---
        text_cols_available = all(c in df.columns for c in self.TEXT_COLUMNS)
        if text_cols_available:
            self._fit_text_tfidf(df)
            text_tfidf_df = self._transform_text_tfidf(df)
            result = pd.concat([result, text_tfidf_df], axis=1)

            text_stats = self._add_text_stats(df)
            result = pd.concat([result, text_stats], axis=1)

        # --- Текстовые колонки как есть (для CatBoost text_features) ---
        if text_cols_available:
            for col in self.TEXT_COLUMNS:
                if col in df.columns:
                    result[col] = df[col].fillna('').astype(str)
            logger.info(
                f"Текстовые колонки сохранены для CatBoost text_features"
            )

        n_total = len(result.columns)
        logger.info(f"Генерация признаков завершена: {n_total} признаков")
        return result, self.skill_columns

    def transform(self, df):
        """Применение обученных трансформаций к новым данным."""
        result = pd.DataFrame(index=df.index)

        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        for col in self.BINARY_COLUMNS:
            if col in df.columns:
                result[col] = df[col].fillna(0).astype(int)

        for col in self.CAT_COLUMNS:
            if col in df.columns:
                result[col] = df[col].fillna('Не указано').astype(str)

        enc_df = self._transform_label_encoders(df)
        result = pd.concat([result, enc_df], axis=1)

        title_df = self._add_title_features(df)
        result = pd.concat([result, title_df], axis=1)

        employer_df = self._add_employer_features(df)
        result = pd.concat([result, employer_df], axis=1)

        if self.tfidf_vectorizer is not None and 'skills_list' in df.columns:
            tfidf_df = self._transform_tfidf(df)
            result = pd.concat([result, tfidf_df], axis=1)

        if self.top_skills and 'skills_list' in df.columns:
            skills_df = self._transform_top_skills(df)
            result = pd.concat([result, skills_df], axis=1)

        text_cols_available = all(c in df.columns for c in self.TEXT_COLUMNS)
        if self.text_vectorizer is not None and text_cols_available:
            text_tfidf_df = self._transform_text_tfidf(df)
            result = pd.concat([result, text_tfidf_df], axis=1)
            text_stats = self._add_text_stats(df)
            result = pd.concat([result, text_stats], axis=1)

        # Текстовые колонки для CatBoost
        if text_cols_available:
            for col in self.TEXT_COLUMNS:
                if col in df.columns:
                    result[col] = df[col].fillna('').astype(str)

        logger.info(f"Признаки применены: {len(result.columns)} колонок")
        return result, self.skill_columns 