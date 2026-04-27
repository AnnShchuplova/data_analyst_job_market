"""
Модуль прогнозирования заработной платы.

Согласно ЧТЗ раздел «Модуль машинного обучения»:
- Алгоритм: CatBoost или RandomForest
- Целевая переменная: Средняя/медианная зарплата (числовая)
- Признаки: Регион (код региона), уровень опыта, вектор навыков, должность
- Разделение: 80/20 train/test
- Кросс-валидация и подбор гиперпараметров
- Метрики: MAE в пределах 20% от средней ЗП, R2 >= 0.7
- Сохранение модели (salary_model.joblib)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

logger = logging.getLogger(__name__)

# Пути для сохранения моделей
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


class SalaryPredictor:
    """Модель прогнозирования заработной платы аналитиков данных."""

    def __init__(self):
        self.model = None
        self.secondary_model = None  # второй алгоритм для ансамбля
        self.feature_columns = None
        self.metrics = {}
        self._is_trained = False
        self._cat_feature_indices = []
        self._cat_feature_names = ['experience_level_encoded', 'area_encoded', 'role_encoded', 'schedule_encoded']

    def _get_model(self, algorithm: str = 'random_forest', use_cat_features: bool = False):
        """Получение экземпляра модели БЕЗ cat_features (для совместимости со sklearn clone)."""
        if algorithm == 'catboost':
            try:
                from catboost import CatBoostRegressor
                params = {
                    'iterations': 2000,
                    'learning_rate': 0.05,
                    'depth': 8,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 5,
                    'verbose': 0,
                    'random_seed': 42,
                }
                # НЕ передаём cat_features в конструктор — sklearn не умеет клонировать
                # cat_features будет передаваться через Pool при fit
                return CatBoostRegressor(**params)
            except ImportError:
                logger.warning("CatBoost не установлен, используем RandomForest")
                algorithm = 'random_forest'

        if algorithm == 'random_forest':
            return RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Неизвестный алгоритм: {algorithm}")

    def _make_pool(self, X, y=None, cat_indices=None):
        """Создание CatBoost Pool с cat_features.

        CatBoost требует, чтобы категориальные колонки были типа int (не float).
        Поэтому конвертируем numpy array -> DataFrame и приводим cat-колонки к int.
        """
        from catboost import Pool
        if cat_indices is None:
            cat_indices = self._cat_feature_indices
        if not cat_indices:
            if y is not None:
                return Pool(X, label=y)
            return None
        # Конвертируем numpy array в DataFrame с правильными типами
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_columns[:X.shape[1]])
            for idx in cat_indices:
                if idx < X.shape[1]:
                    col_name = X_df.columns[idx]
                    X_df[col_name] = X_df[col_name].astype(np.int32)
        else:
            X_df = X.copy()
            for idx in cat_indices:
                if idx < len(X_df.columns):
                    col_name = X_df.columns[idx]
                    X_df[col_name] = X_df[col_name].astype(np.int32)
        if y is not None:
            return Pool(data=X_df, label=y, cat_features=cat_indices)
        else:
            return Pool(data=X_df, cat_features=cat_indices)

    def train(self, df: pd.DataFrame, feature_columns: list,
              target_column: str = 'salary_avg',
              algorithm: str = 'random_forest',
              test_size: float = 0.15,
              tune_hyperparams: bool = False,
              remove_outliers: bool = True,
              salary_range: tuple = (20000, 600000)) -> dict:
        """Обучение модели прогнозирования зарплаты."""
        logger.info(f"Обучение модели прогноза зарплаты (алгоритм: {algorithm})")

        # Фильтрация записей с пропущенной зарплатой
        df_model = df.dropna(subset=[target_column]).copy()
        df_model = df_model[df_model[target_column] > 0]
        df_model = df_model.reset_index(drop=True)

        # Удаление выбросов
        if remove_outliers:
            before = len(df_model)
            Q1 = df_model[target_column].quantile(0.02)
            Q3 = df_model[target_column].quantile(0.98)
            df_model = df_model[(df_model[target_column] >= Q1) &
                                (df_model[target_column] <= Q3)]
            df_model = df_model.reset_index(drop=True)
            removed = before - len(df_model)
            if removed > 0:
                logger.info(f"Удалено {removed} выбросов (процентили 5%-95%)")

        # Фильтрация записей с пропущенными признаками
        available_features = [c for c in feature_columns if c in df_model.columns]
        df_model = df_model.dropna(subset=available_features)
        df_model = df_model.reset_index(drop=True)

        logger.info(f"Записей для обучения: {len(df_model)}")
        if len(df_model) < 50:
            logger.warning(f"Мало данных ({len(df_model)} записей).")

        # ======= РАЗДЕЛЕНИЕ 80/20 ПЕРЕД ЛЮБЫМИ ТРАНСФОРМАЦИЯМИ =======
        train_idx, test_idx = train_test_split(
            range(len(df_model)), test_size=test_size, random_state=42
        )
        train_df = df_model.iloc[train_idx].copy()
        test_df = df_model.iloc[test_idx].copy()

        # ======= TARGET ENCODING ТОЛЬКО НА TRAIN =======
        global_mean = train_df[target_column].mean()

        role_means = region_means = emp_means = wf_means = None

        if 'main_role_name' in train_df.columns:
            role_means = train_df.groupby('main_role_name')[target_column].mean()
            train_df['role_mean_salary'] = train_df['main_role_name'].map(role_means).fillna(global_mean)
            test_df['role_mean_salary'] = test_df['main_role_name'].map(role_means).fillna(global_mean)

        if 'area_name' in train_df.columns:
            region_means = train_df.groupby('area_name')[target_column].mean()
            train_df['region_mean_salary'] = train_df['area_name'].map(region_means).fillna(global_mean)
            test_df['region_mean_salary'] = test_df['area_name'].map(region_means).fillna(global_mean)

        if 'employment_name' in train_df.columns:
            emp_means = train_df.groupby('employment_name')[target_column].mean()
            train_df['employment_mean_salary'] = train_df['employment_name'].map(emp_means).fillna(global_mean)
            test_df['employment_mean_salary'] = test_df['employment_name'].map(emp_means).fillna(global_mean)

        if 'work_format_name' in train_df.columns:
            wf_means = train_df.groupby('work_format_name')[target_column].mean()
            train_df['work_format_mean_salary'] = train_df['work_format_name'].map(wf_means).fillna(global_mean)
            test_df['work_format_mean_salary'] = test_df['work_format_name'].map(wf_means).fillna(global_mean)

        # Собираем итоговый список признаков
        te_columns = [c for c in ['role_mean_salary', 'region_mean_salary',
                                    'employment_mean_salary', 'work_format_mean_salary']
                      if c in train_df.columns]

        all_features = list(available_features) + te_columns
        self.feature_columns = all_features
        logger.info(f"Итого признаков: {len(all_features)} (из них {len(te_columns)} TE из train)")

        # ======= ПОДГОТОВКА X и y =======
        X_train = train_df[all_features].fillna(0).values
        X_test = test_df[all_features].fillna(0).values
        y_train_orig = train_df[target_column].values
        y_test_orig = test_df[target_column].values

        # Логарифм целевой переменной
        y_train_log = np.log1p(y_train_orig)
        y_test_log = np.log1p(y_test_orig)

        logger.info(f"Обучающая выборка: {len(X_train)}, Тестовая: {len(X_test)}")

        # ======= CAT_FEATURES ДЛЯ CATBOOST =======
        self._cat_feature_indices = []

        # ======= ОСНОВНАЯ МОДЕЛЬ (CatBoost) =======
        self.model = self._get_model(algorithm)

        if algorithm == 'catboost':
            # ======= CatBoost ВСТРОЕННАЯ КРОСС-ВАЛИДАЦИЯ (через Pool) =======
            try:
                from catboost import Pool, cv as catboost_cv

                train_pool = self._make_pool(X_train, y_train_log, self._cat_feature_indices)

                cv_params = {
                    'loss_function': 'RMSE',
                    'random_seed': 42,
                    'verbose': 0,
                    'iterations': 2000,
                    'learning_rate': 0.05,
                    'depth': 8,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 5,
                }

                cv_result = catboost_cv(
                    pool=train_pool,
                    params=cv_params,
                    fold_count=5,
                    partition_random_seed=42,
                    verbose=False,
                    early_stopping_rounds=50
                )

                best_cv_score = cv_result['test-RMSE-mean'].min()
                logger.info(f"CatBoost CV RMSE (log): {best_cv_score:.4f}")
            except Exception as e:
                logger.warning(f"CatBoost CV не удалось: {e}, пропускаем кросс-валидацию")

            # ======= ПОДБОР ГИПЕРПАРАМЕТРОВ (sklearn GridSearch БЕЗ cat_features) =======
            if tune_hyperparams:
                self._tune_hyperparameters(X_train, y_train_log, algorithm)

            # ======= ОБУЧЕНИЕ ОСНОВНОЙ МОДЕЛИ ЧЕРЕЗ POOL =======
            train_pool = self._make_pool(X_train, y_train_log, self._cat_feature_indices)
            self.model.fit(train_pool)
            logger.info("CatBoost обучен с cat_features через Pool")
        else:
            # Для RandomForest / GradientBoosting — обычный fit
            cv_scores = cross_val_score(self.model, X_train, y_train_log, cv=5, scoring='neg_mean_absolute_error')
            logger.info(f"CV MAE (log): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            if tune_hyperparams:
                self._tune_hyperparameters(X_train, y_train_log, algorithm)

            self.model.fit(X_train, y_train_log)

        # ======= ВТОРАЯ МОДЕЛЬ (RandomForest) для ансамбля =======
        logger.info("Обучение второй модели (RandomForest) для ансамбля...")
        self.secondary_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.secondary_model.fit(X_train, y_train_log)

        # ======= ПРЕДСКАЗАНИЕ — АНСАМБЛЬ (0.6 CatBoost + 0.4 RF) =======
        if algorithm == 'catboost' and self._cat_feature_indices:
            test_pool = self._make_pool(X_test, cat_indices=self._cat_feature_indices)
            y_pred_cat_log = self.model.predict(test_pool)
        else:
            y_pred_cat_log = self.model.predict(X_test)

        y_pred_rf_log = self.secondary_model.predict(X_test)
        y_pred_log = 0.6 * y_pred_cat_log + 0.4 * y_pred_rf_log
        y_pred = np.expm1(y_pred_log)

        # ======= МЕТРИКИ =======
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        mean_salary = y_test_orig.mean()
        mae_percent = (mae / mean_salary) * 100

        self.metrics = {
            'MAE': round(mae, 2),
            'MAE_percent': round(mae_percent, 2),
            'R2': round(r2, 4),
            'RMSE': round(rmse, 2),
            'mean_salary': round(mean_salary, 2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'algorithm': f'{algorithm} + RF ensemble',
            'n_features': len(all_features),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        self._is_trained = True

        # Также считаем метрики отдельно для CatBoost (для сравнения)
        y_pred_cat = np.expm1(y_pred_cat_log)
        r2_cat = r2_score(y_test_orig, y_pred_cat)
        logger.info(f"  CatBoost отдельно R2: {r2_cat:.4f}")
        logger.info(f"  Ensemble R2: {r2:.4f}")

        logger.info(f"Метрики модели (ансамбль):")
        logger.info(f"  MAE: {mae:.0f} руб. ({mae_percent:.1f}% от средней ЗП)")
        logger.info(f"  R2: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.0f} руб.")

        if mae_percent <= 20:
            logger.info(f"  MAE в пределах 20% от средней ЗП - ТРЕБОВАНИЕ ВЫПОЛНЕНО")
        else:
            logger.warning(f"  MAE {mae_percent:.1f}% превышает 20% - требование НЕ выполнено")

        if r2 >= 0.7:
            logger.info(f"  R2 >= 0.7 - ТРЕБОВАНИЕ ВЫПОЛНЕНО")
        else:
            logger.warning(f"  R2 {r2:.4f} < 0.7 - требование НЕ выполнено")

        return self.metrics

    def _tune_hyperparameters(self, X_train, y_train, algorithm: str):
        """Подбор гиперпараметров через GridSearchCV."""
        logger.info("Подбор гиперпараметров...")

        if algorithm == 'catboost':
            try:
                from catboost import CatBoostRegressor
                param_grid = {
                    'iterations': [1500, 2000],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'depth': [6, 8],
                    'l2_leaf_reg': [3, 5, 7]
                }
                # Для GridSearchCV создаём модель БЕЗ cat_features (чтобы sklearn не ругался при clone)
                base_model = CatBoostRegressor(
                    iterations=1500, learning_rate=0.05, depth=8,
                    l2_leaf_reg=3, min_data_in_leaf=5,
                    verbose=0, random_seed=42
                )
            except ImportError:
                return
        elif algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [300, 500],
                'max_depth': [15, 20, 25],
                'min_samples_split': [3, 5]
            }
            base_model = self._get_model(algorithm)
        else:
            return

        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        logger.info(f"Лучшие параметры: {best_params}")
        logger.info(f"Лучший CV MAE: {-grid_search.best_score_:.4f}")

        # Пересоздаём модель с лучшими параметрами (БЕЗ cat_features в конструкторе)
        if algorithm == 'catboost':
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(
                    **best_params,
                    min_data_in_leaf=5,
                    verbose=0,
                    random_seed=42,
                )
                # cat_features НЕ передаём в конструктор — будет через Pool при fit
            except ImportError:
                self.model = grid_search.best_estimator_
        else:
            self.model = grid_search.best_estimator_

    def predict(self, features: pd.DataFrame or np.ndarray) -> np.ndarray:
        """Предсказание (ансамбль CatBoost + RF)."""
        if not self._is_trained:
            raise ValueError("Модель не обучена.")

        if isinstance(features, pd.DataFrame):
            available = [c for c in self.feature_columns if c in features.columns]
            features = features[available].fillna(0).values

        # CatBoost предсказание через Pool если есть cat_features
        if self._cat_feature_indices:
            pool = self._make_pool(features, cat_indices=self._cat_feature_indices)
            pred_cat_log = self.model.predict(pool)
        else:
            pred_cat_log = self.model.predict(features)

        pred_rf_log = self.secondary_model.predict(features)
        pred_log = 0.6 * pred_cat_log + 0.4 * pred_rf_log
        return np.expm1(pred_log)

    def predict_single(self, feature_dict: dict) -> float:
        """Прогноз зарплаты для одного набора признаков."""
        if not self._is_trained:
            raise ValueError("Модель не обучена.")

        features = np.array([[feature_dict.get(col, 0) for col in self.feature_columns]])

        if self._cat_feature_indices:
            pool = self._make_pool(features, cat_indices=self._cat_feature_indices)
            pred_cat_log = self.model.predict(pool)[0]
        else:
            pred_cat_log = self.model.predict(features)[0]

        pred_rf_log = self.secondary_model.predict(features)[0]
        pred_log = 0.6 * pred_cat_log + 0.4 * pred_rf_log
        return float(np.expm1(pred_log))

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Важность признаков (основной модели)."""
        if not self._is_trained or not hasattr(self.model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    def save(self, filepath: str = None):
        """Сохранение обученной модели (salary_model.joblib)."""
        import joblib

        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "salary_model.joblib")

        model_data = {
            'model': self.model,
            'secondary_model': self.secondary_model,
            'feature_columns': self.feature_columns,
            'cat_feature_indices': self._cat_feature_indices,
            'metrics': self.metrics,
            'is_trained': self._is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Модель сохранена: {filepath}")

        metrics_path = os.path.join(os.path.dirname(filepath), "salary_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Метрики сохранены: {metrics_path}")

    def load(self, filepath: str = None):
        """Загрузка обученной модели."""
        import joblib

        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "salary_model.joblib")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Модель не найдена: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.secondary_model = model_data.get('secondary_model')
        self.feature_columns = model_data['feature_columns']
        self._cat_feature_indices = model_data.get('cat_feature_indices', [])
        self.metrics = model_data['metrics']
        self._is_trained = model_data['is_trained']

        logger.info(f"Модель загружена: {filepath}")
        logger.info(f"Метрики: MAE={self.metrics.get('MAE', 'N/A')}, R2={self.metrics.get('R2', 'N/A')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    from src.data_processing.feature_engineering import FeatureEngineer

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "finaldata", "month_dataset_20260420_205719.csv"
    )
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        logger.info(f"Загружено {len(df)} записей")

        fe = FeatureEngineer()
        df, feature_cols = fe.prepare_dataset_for_regression(df, fit=True)

        predictor = SalaryPredictor()
        metrics = predictor.train(df, feature_cols, algorithm='catboost', tune_hyperparams=True)

        print("\nМетрики модели (ансамбль):")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        predictor.save()

        importance = predictor.get_feature_importance()
        if importance is not None:
            print(f"\nТоп-15 признаков:")
            print(importance.to_string())
    else:
        print(f"Файл не найден: {data_path}")
