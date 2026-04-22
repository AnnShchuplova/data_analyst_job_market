"""
Модуль генерации признаков для ML-моделей.

Согласно ЧТЗ:
- Создание категориального признака уровня опыта: Junior, Middle, Senior
- Векторизация навыков (TF-IDF или one-hot encoding)
- Подготовка датасета для ML
"""

import pandas as pd
import numpy as np
import logging
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Генерация и кодирование признаков для ML-моделей."""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=150, min_df=2, max_df=0.90)
        self.mlb = None
        self.experience_le = LabelEncoder()
        self.area_le = LabelEncoder()
        self.role_le = LabelEncoder()
        self.schedule_le = LabelEncoder()
        self._fitted = False
        self._skill_columns = None

    def create_experience_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание категориального признака уровня опыта: Junior, Middle, Senior."""
        df = df.copy()

        if 'experience_id' in df.columns:
            level_map = {
                'noExperience': 'Junior',
                'between1And3': 'Middle',
                'between3And6': 'Senior',
                'moreThan6': 'Senior'
            }
            df['experience_level'] = df['experience_id'].map(level_map)
        elif 'experience_name' in df.columns:
            level_map = {
                'Нет опыта': 'Junior',
                'От 1 года до 3 лет': 'Middle',
                'От 3 до 6 лет': 'Senior',
                'Более 6 лет': 'Senior'
            }
            df['experience_level'] = df['experience_name'].map(level_map)
        else:
            logger.warning("Колонка experience_id/experience_name не найдена")
            df['experience_level'] = 'Unknown'

        level_counts = df['experience_level'].value_counts()
        logger.info(f"Уровни опыта: {level_counts.to_dict()}")

        return df

    def prepare_skills_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Преобразование списка навыков в текстовую строку для TF-IDF."""
        df = df.copy()

        def _skills_to_text(val):
            if isinstance(val, list) and len(val) > 0:
                return ' '.join(val)
            if isinstance(val, str) and val.strip():
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return ' '.join(parsed)
                except (ValueError, SyntaxError):
                    pass
            return ''

        if 'skills_list' in df.columns:
            df['skills_text'] = df['skills_list'].apply(_skills_to_text)

        text_cols = ['requirement', 'responsibility', 'name']
        available_text_cols = [c for c in text_cols if c in df.columns]

        if available_text_cols:
            for col in available_text_cols:
                df[col] = df[col].fillna('').astype(str)

            def _combine_text(row):
                skills = row.get('skills_text', '')
                if skills and skills.strip():
                    return skills
                parts = [str(row.get(c, '')) for c in available_text_cols]
                combined = ' '.join(parts)
                return combined

            df['skills_text'] = df.apply(_combine_text, axis=1)
        elif 'skills_text' not in df.columns:
            logger.warning("Колонки skills_list и текстовых полей не найдены")
            df['skills_text'] = ''

        return df

    def fit_transform_skills_tfidf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучение TF-IDF векторизатора на навыках и трансформация."""
        df = df.copy()
        df = self.prepare_skills_text(df)

        non_empty_mask = df['skills_text'].str.len() > 0
        non_empty_count = non_empty_mask.sum()

        if non_empty_count == 0:
            logger.warning("Нет навыков для TF-IDF векторизации")
            return df

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df.loc[non_empty_mask, 'skills_text'])
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'skill_tfidf_{col}' for col in tfidf_feature_names],
            index=df.loc[non_empty_mask].index
        )

        zeros_df = pd.DataFrame(0.0, index=df.index, columns=tfidf_df.columns)
        zeros_df.loc[non_empty_mask] = tfidf_df
        df = pd.concat([df, zeros_df], axis=1)

        self._fitted = True
        self._skill_columns = list(tfidf_df.columns)

        logger.info(f"TF-IDF: создано {len(tfidf_feature_names)} признаков из {non_empty_count} записей")

        return df

    def transform_skills_tfidf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Трансформация новых данных с обученным TF-IDF векторизатором."""
        if not self._fitted:
            raise ValueError("Векторизатор не обучен. Сначала вызовите fit_transform_skills_tfidf().")

        df = df.copy()
        df = self.prepare_skills_text(df)

        non_empty_mask = df['skills_text'].str.len() > 0

        if non_empty_mask.sum() == 0:
            for col in self._skill_columns:
                df[col] = 0.0
            return df

        tfidf_matrix = self.tfidf_vectorizer.transform(df.loc[non_empty_mask, 'skills_text'])

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self._skill_columns,
            index=df.loc[non_empty_mask].index
        )

        for col in self._skill_columns:
            df[col] = 0.0
        df.loc[non_empty_mask, self._skill_columns] = tfidf_df

        return df

    def _add_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление бинарных и числовых признаков из текста."""
        df = df.copy()
        req_text = df['requirement'].fillna('').astype(str).str.lower()
        name_text = df['name'].fillna('').astype(str).str.lower()
        resp_text = df.get('responsibility', pd.Series('', index=df.index)).fillna('').astype(str).str.lower()

        # Бинарные признаки — технологии
        df['has_sql'] = req_text.str.contains('sql').astype(int)
        df['has_python'] = req_text.str.contains('python|питон').astype(int)
        df['has_bi'] = req_text.str.contains('power bi|tableau|bi |дашборд|dashboard').astype(int)
        df['has_ml'] = req_text.str.contains('машинн|machine learning|нейронн|deep learning').astype(int)
        df['has_stats'] = req_text.str.contains('статистик|a/b|ab тест|математик').astype(int)
        df['has_etl'] = req_text.str.contains('etl|airflow|data pipeline|elt').astype(int)
        df['has_cloud'] = req_text.str.contains('aws|gcp|azure|облак|yandex cloud').astype(int)
        df['has_excel'] = req_text.str.contains('excel').astype(int)
        df['has_java'] = req_text.str.contains('java|kafka|spark|hadoop').astype(int)
        df['has_git'] = req_text.str.contains('git|github|gitlab').astype(int)
        df['has_db'] = req_text.str.contains('postgresql|postgres|mysql|clickhouse|mongo|база данных').astype(int)
        df['has_team_lead'] = req_text.str.contains('руковод|управлен|команд|lead').astype(int)
        df['has_communication'] = resp_text.str.contains('коммуникац|переговор|взаимодейств|стейкхолдер').astype(int)
        df['has_english'] = req_text.str.contains('english|английск').astype(int)
        df['has_presentations'] = resp_text.str.contains('презентац|отчёт|отчет|доклад').astype(int)

        # Количество навыков
        df['key_skills_total'] = df[['has_sql', 'has_python', 'has_bi', 'has_ml',
                                       'has_stats', 'has_etl', 'has_cloud', 'has_java',
                                       'has_db', 'has_git', 'has_english']].sum(axis=1)

        # Технические vs бизнес навыки
        df['tech_skills_count'] = df[['has_sql', 'has_python', 'has_ml', 'has_etl',
                                       'has_cloud', 'has_java', 'has_db', 'has_git']].sum(axis=1)
        df['biz_skills_count'] = df[['has_bi', 'has_stats', 'has_excel',
                                      'has_team_lead', 'has_communication',
                                      'has_presentations', 'has_english']].sum(axis=1)

        # Из названия вакансии
        df['name_has_senior'] = name_text.str.contains('senior|lead|principal|руководитель|главный|ведущий').astype(int)
        df['name_has_junior'] = name_text.str.contains('junior|младший|стажер|intern|ученик').astype(int)
        df['name_has_middle'] = name_text.str.contains('middle').astype(int)

        # Формат работы
        if 'schedule_name' in df.columns:
            df['is_remote'] = (df['schedule_name'].str.contains('Удаленн', na=False)).astype(int)
            df['is_hybrid'] = (df['schedule_name'].str.contains('Гибрид', na=False)).astype(int)
        else:
            df['is_remote'] = 0
            df['is_hybrid'] = 0

        # Опыт
        df['is_senior'] = (df['experience_id'].isin(['between3And6', 'moreThan6'])).astype(int)
        df['is_junior_exp'] = (df['experience_id'].isin(['noExperience'])).astype(int)

        # Текстовые признаки
        df['text_length'] = req_text.str.len()
        df['skills_count'] = df['skills_list'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление признаков взаимодействия."""
        df = df.copy()

        # Опыт × технические навыки
        if 'tech_skills_count' in df.columns:
            df['exp_x_tech'] = df.get('is_senior', 0) * df['tech_skills_count']

        # Удалёнка × senior
        if 'is_remote' in df.columns:
            df['remote_x_senior'] = df.get('is_remote', 0) * df.get('is_senior', 0)
            df['remote_x_tech'] = df.get('is_remote', 0) * df.get('key_skills_total', 0)

        # Навыки × опыт (junior с 5 навыками ценнее middle с 2)
        if 'skills_count' in df.columns:
            df['skills_x_exp'] = df['skills_count'] * df.get('is_senior', 0)

        # Москва/Питер бонус (encode как is_top_city)
        if 'area_name' in df.columns:
            top_cities = ['Москва', 'Санкт-Петербург']
            df['is_top_city'] = df['area_name'].isin(top_cities).astype(int)
        else:
            df['is_top_city'] = 0

        return df

    def fit_transform_experience_level_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование experience_level."""
        df = df.copy()

        if 'experience_level' not in df.columns:
            df = self.create_experience_level(df)

        df['experience_level_encoded'] = self.experience_le.fit_transform(
            df['experience_level'].astype(str)
        )

        logger.info(f"Кодирование опыта: {dict(zip(self.experience_le.classes_, self.experience_le.transform(self.experience_le.classes_)))}")

        return df

    def transform_experience_level_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование experience_level для новых данных."""
        df = df.copy()

        if 'experience_level' not in df.columns:
            df = self.create_experience_level(df)

        known_classes = set(self.experience_le.classes_)
        df['experience_level'] = df['experience_level'].astype(str)
        df['experience_level'] = df['experience_level'].apply(lambda x: x if x in known_classes else 'Middle')
        df['experience_level_encoded'] = self.experience_le.transform(df['experience_level'])

        return df

    def fit_transform_area_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование региона."""
        df = df.copy()

        area_col = 'area_name' if 'area_name' in df.columns else 'area'
        if area_col not in df.columns:
            logger.warning("Колонка area_name/area не найдена")
            df['area_encoded'] = 0
            return df

        if area_col == 'area_name':
            df['area_encoded'] = self.area_le.fit_transform(df[area_col].fillna('Не указан').astype(str))
        else:
            df['area_encoded'] = self.area_le.fit_transform(
                df[area_col].apply(lambda x: x.get('name', 'Не указан') if isinstance(x, dict) else 'Не указан').astype(str)
            )

        logger.info(f"Кодирование регионов: {len(self.area_le.classes_)} уникальных значений")

        return df

    def transform_area_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование региона для новых данных."""
        df = df.copy()

        area_col = 'area_name' if 'area_name' in df.columns else 'area'
        if area_col not in df.columns:
            df['area_encoded'] = 0
            return df

        known_classes = set(self.area_le.classes_)
        if area_col == 'area_name':
            area_values = df[area_col].fillna('Не указан').astype(str)
            area_values = area_values.apply(lambda x: x if x in known_classes else 'Не указан')
            df['area_encoded'] = self.area_le.transform(area_values)
        else:
            df['area_encoded'] = self.area_le.transform(
                df[area_col].apply(lambda x: x.get('name', 'Не указан') if isinstance(x, dict) else 'Не указан').astype(str)
            )

        return df

    def fit_transform_role_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование роли."""
        df = df.copy()

        if 'main_role_name' not in df.columns:
            logger.warning("Колонка main_role_name не найдена")
            df['role_encoded'] = 0
            return df

        df['role_encoded'] = self.role_le.fit_transform(df['main_role_name'].fillna('Не указана').astype(str))
        logger.info(f"Кодирование ролей: {len(self.role_le.classes_)} уникальных значений")

        return df

    def transform_role_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование роли для новых данных."""
        df = df.copy()

        if 'main_role_name' not in df.columns:
            df['role_encoded'] = 0
            return df

        known_classes = set(self.role_le.classes_)
        values = df['main_role_name'].fillna('Не указана').astype(str)
        values = values.apply(lambda x: x if x in known_classes else 'Не указана')
        df['role_encoded'] = self.role_le.transform(values)

        return df

    def fit_transform_schedule_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование графика работы."""
        df = df.copy()

        if 'schedule_name' not in df.columns:
            logger.warning("Колонка schedule_name не найдена")
            df['schedule_encoded'] = 0
            return df

        df['schedule_encoded'] = self.schedule_le.fit_transform(df['schedule_name'].fillna('Не указан').astype(str))
        logger.info(f"Кодирование графика: {len(self.schedule_le.classes_)} уникальных значений")

        return df

    def transform_schedule_encoded(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование графика для новых данных."""
        df = df.copy()

        if 'schedule_name' not in df.columns:
            df['schedule_encoded'] = 0
            return df

        known_classes = set(self.schedule_le.classes_)
        values = df['schedule_name'].fillna('Не указан').astype(str)
        values = values.apply(lambda x: x if x in known_classes else 'Не указан')
        df['schedule_encoded'] = self.schedule_le.transform(values)

        return df

    def prepare_dataset_for_regression(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Подготовка полного набора признаков для задачи регрессии (прогноз зарплаты).

        Target encoding выполняется в SalaryPredictor.train() после split,
        чтобы избежать утечки данных из test в train.
        """
        if fit:
            df = self.create_experience_level(df)
            df = self.fit_transform_skills_tfidf(df)
            df = self.fit_transform_experience_level_encoded(df)
            df = self.fit_transform_area_encoded(df)
            df = self.fit_transform_role_encoded(df)
            df = self.fit_transform_schedule_encoded(df)
        else:
            df = self.create_experience_level(df)
            df = self.transform_skills_tfidf(df)
            df = self.transform_experience_level_encoded(df)
            df = self.transform_area_encoded(df)
            df = self.transform_role_encoded(df)
            df = self.transform_schedule_encoded(df)

        # Добавляем бинарные и interaction-признаки
        df = self._add_binary_features(df)
        df = self._add_interaction_features(df)

        # Формируем список признаков
        feature_cols = [
            'experience_level_encoded', 'area_encoded', 'role_encoded', 'schedule_encoded',
        ]
        if 'min_experience_years' in df.columns:
            feature_cols.append('min_experience_years')
        if 'skills_count' in df.columns:
            feature_cols.append('skills_count')

        # Все числовые признаки
        numeric_features = [
            'has_sql', 'has_python', 'has_bi', 'has_ml', 'has_stats',
            'has_etl', 'has_cloud', 'has_excel',
            'has_java', 'has_git', 'has_db', 'has_team_lead', 'has_communication',
            'has_english', 'has_presentations',
            'key_skills_total', 'tech_skills_count', 'biz_skills_count',
            'name_has_senior', 'name_has_junior', 'name_has_middle',
            'is_remote', 'is_hybrid', 'is_senior', 'is_junior_exp',
            'is_top_city',
            'text_length',
            # Interaction
            'exp_x_tech', 'remote_x_senior', 'remote_x_tech', 'skills_x_exp',
        ]
        for nf in numeric_features:
            if nf in df.columns:
                feature_cols.append(nf)

        # TF-IDF признаки
        if self._skill_columns:
            feature_cols.extend(self._skill_columns)

        # Убираем nonexistent columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Заполняем NaN нулями
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)

        logger.info(f"Подготовлено {len(feature_cols)} признаков для регрессии")

        return df, feature_cols


if __name__ == "__main__":
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "finaldata", "month_dataset_20260420_205719.csv"
    )

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        fe = FeatureEngineer()
        df = fe.create_experience_level(df)
        print(f"Уровни опыта:\n{df['experience_level'].value_counts()}")

        df, feature_cols = fe.prepare_dataset_for_regression(df, fit=True)
        print(f"\nПризнаков для регрессии: {len(feature_cols)}")
    else:
        print(f"Файл не найден: {data_path}")
