"""
Модуль ML-моделей для предсказания зарплаты.
Включает RandomForest, CatBoost, GradientBoosting и ансамбль.

Ключевые правила:
  - Целевое кодирование вычисляется ТОЛЬКО после разделения train/test
  - CatBoost получает нативные cat_features + text_features
  - RandomForest/GradientBoosting используют ordinal encoding + TF-IDF числа
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

logger = logging.getLogger(__name__)


class SalaryPredictor:
    """Предсказатель зарплаты на основе ML-моделей."""

    def __init__(self):
        self.models = {}
        self.best_model_name = None
        self.feature_engineer = None

        self.cat_columns = [
            'experience_name', 'area_name', 'main_role_name',
            'schedule_name', 'work_format_name', 'employment_name'
        ]

        self.text_columns = ['requirement', 'responsibility']

        self.te_min_samples = 5

        self.encoded_columns = [f'{col}_enc' for col in self.cat_columns]

        self.target_encoding_maps = {}
        self.te_col_mapping = {}
        self.te_fallback = None

    # -------------------------------------------------------
    # Целевое кодирование — БЕЗ утечки, со сглаживанием
    # -------------------------------------------------------
    def _add_target_encoding(self, X_train, X_test, y_train):
        combined = X_train.copy()
        combined['_target'] = y_train

        te_configs = [
            ('main_role_name', 'role_mean_salary'),
            ('area_name', 'region_mean_salary'),
            ('experience_name', 'exp_mean_salary'),
            ('schedule_name', 'schedule_mean_salary'),
            ('work_format_name', 'workformat_mean_salary'),
            ('employment_name', 'employment_mean_salary'),
        ]

        self.target_encoding_maps = {}
        self.te_col_mapping = {}
        info_parts = []
        global_mean = y_train.mean()

        for cat_col, te_col in te_configs:
            if cat_col in X_train.columns:
                group_stats = combined.groupby(cat_col)['_target'].agg(
                    ['mean', 'count']
                )
                smoothing = group_stats['count'] / (
                    group_stats['count'] + self.te_min_samples
                )
                smoothed_means = (
                    smoothing * group_stats['mean'] +
                    (1 - smoothing) * global_mean
                )
                self.target_encoding_maps[te_col] = smoothed_means
                self.te_col_mapping[te_col] = cat_col

                X_train[te_col] = X_train[cat_col].map(smoothed_means)
                X_test[te_col] = X_test[cat_col].map(smoothed_means)

                info_parts.append(f"{cat_col}={len(smoothed_means)}")

        self.te_fallback = float(y_train.mean())

        for _, te_col in te_configs:
            if te_col in X_train.columns:
                X_train[te_col] = X_train[te_col].fillna(self.te_fallback)
                X_test[te_col] = X_test[te_col].fillna(self.te_fallback)

        logger.info(
            f"Целевое кодирование (сглаженное, только по train): "
            f"{', '.join(info_parts)}, fallback={self.te_fallback:.0f}"
        )

        return X_train, X_test

    # -------------------------------------------------------
    # Подготовка признаков для RandomForest / GradientBoosting
    # -------------------------------------------------------
    def _prepare_rf_features(self, X):
        rf = X.copy()
        for col in self.cat_columns:
            if col in rf.columns:
                rf.drop(col, axis=1, inplace=True)
        for col in self.text_columns:
            if col in rf.columns:
                rf.drop(col, axis=1, inplace=True)
        rf = rf.fillna(0)
        return rf

    # -------------------------------------------------------
    # Индексы категориальных и текстовых признаков для CatBoost
    # -------------------------------------------------------
    def _get_cat_feature_indices(self, X):
        indices = []
        for col in self.cat_columns:
            if col in X.columns:
                indices.append(X.columns.get_loc(col))
        return indices

    def _get_text_feature_indices(self, X):
        indices = []
        for col in self.text_columns:
            if col in X.columns:
                indices.append(X.columns.get_loc(col))
        return indices

    # -------------------------------------------------------
    # Метрики
    # -------------------------------------------------------
    def _evaluate(self, y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mean_salary = y_true.mean()
        mae_pct = (mae / mean_salary) * 100

        logger.info(
            f"  [{model_name}] MAE={mae:.0f} ({mae_pct:.1f}%), "
            f"RMSE={rmse:.0f}, R²={r2:.4f}, MAPE={mape:.1f}%"
        )
        return {
            'MAE': mae,
            'MAE%': mae_pct,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

    # -------------------------------------------------------
    # Кросс-валидация
    # -------------------------------------------------------
    def _cross_validate(self, model, X, y, model_name, cv=5):
        try:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring='r2', n_jobs=-1
            )
            mean_r2 = scores.mean()
            std_r2 = scores.std()
            logger.info(
                f"  [{model_name}] Кросс-валидация R²: "
                f"{mean_r2:.4f} ± {std_r2:.4f}"
            )
            return mean_r2
        except Exception as e:
            logger.warning(
                f"  [{model_name}] Кросс-валидация не удалась: {e}"
            )
            return None

    # -------------------------------------------------------
    # Важность признаков
    # -------------------------------------------------------
    def _log_feature_importance(self, model, columns, model_name, top_n=15):
        if not hasattr(model, 'feature_importances_'):
            return
        importances = pd.Series(model.feature_importances_, index=columns)
        importances = importances.sort_values(ascending=False).head(top_n)
        logger.info(f"  Топ-{top_n} признаков [{model_name}]:")
        for feat, imp in importances.items():
            logger.info(f"    {feat}: {imp:.4f}")

    # -------------------------------------------------------
    # ОБУЧЕНИЕ
    # -------------------------------------------------------
    def train(self, X, y, top_skill_cols=None, test_size=0.2, random_state=42):
        logger.info(f"=== Обучение моделей ===")
        logger.info(f"  Данных: {len(X)} строк, {len(X.columns)} признаков")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

        X_train, X_test = self._add_target_encoding(X_train, X_test, y_train)

        X_train_rf = self._prepare_rf_features(X_train)
        X_test_rf = self._prepare_rf_features(X_test)
        logger.info(
            f"  RF/GB-признаков: {len(X_train_rf.columns)} "
            f"(numeric only, без cat/text строк)"
        )

        all_metrics = {}

        # =============================================
        # 1. RandomForest
        # =============================================
        logger.info("--- RandomForest ---")

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )

        cv_r2 = self._cross_validate(rf, X_train_rf, y_train, 'RandomForest')
        rf.fit(X_train_rf, y_train)
        rf_pred = rf.predict(X_test_rf)

        rf_metrics = self._evaluate(y_test, rf_pred, 'RandomForest')
        rf_metrics['CV_R2'] = cv_r2
        all_metrics['RandomForest'] = rf_metrics

        self._log_feature_importance(rf, X_train_rf.columns, 'RandomForest')
        self.models['random_forest'] = rf
        self.models['_rf_columns'] = list(X_train_rf.columns)

        # =============================================
        # 2. CatBoost (cat_features + text_features)
        # =============================================
        logger.info("--- CatBoost (с нативными text_features) ---")

        cat_indices = self._get_cat_feature_indices(X_train)
        text_indices = self._get_text_feature_indices(X_train)

        logger.info(
            f"  CatBoost: cat_features={cat_indices}, "
            f"text_features={text_indices}"
        )

        train_pool = Pool(
            X_train, y_train,
            cat_features=cat_indices,
            text_features=text_indices
        )
        test_pool = Pool(
            X_test, y_test,
            cat_features=cat_indices,
            text_features=text_indices
        )

        cb = CatBoostRegressor(
            iterations=2000,
            depth=7,
            learning_rate=0.02,
            l2_leaf_reg=5,
            bagging_temperature=0.5,
            random_strength=1.0,
            random_seed=random_state,
            verbose=500,
            early_stopping_rounds=100
        )

        logger.info("  [CatBoost] Кросс-валидация пропущена (используется early_stopping)")
        cv_r2 = None
        cb.fit(train_pool, eval_set=test_pool)
        cb_pred = cb.predict(test_pool)

        cb_metrics = self._evaluate(y_test, cb_pred, 'CatBoost')
        cb_metrics['CV_R2'] = cv_r2
        all_metrics['CatBoost'] = cb_metrics

        self._log_feature_importance(cb, X_train.columns, 'CatBoost')
        self.models['catboost'] = cb
        self.models['_cat_columns'] = list(X_train.columns)
        self.models['_cat_indices'] = cat_indices
        self.models['_text_indices'] = text_indices

        # =============================================
        # 3. GradientBoosting
        # =============================================
        logger.info("--- GradientBoosting ---")

        gb = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        )

        cv_r2 = self._cross_validate(gb, X_train_rf, y_train, 'GradientBoosting')
        gb.fit(X_train_rf, y_train)
        gb_pred = gb.predict(X_test_rf)

        gb_metrics = self._evaluate(y_test, gb_pred, 'GradientBoosting')
        gb_metrics['CV_R2'] = cv_r2
        all_metrics['GradientBoosting'] = gb_metrics

        self._log_feature_importance(gb, X_train_rf.columns, 'GradientBoosting')
        self.models['gradient_boosting'] = gb

        # =============================================
        # 4. Ансамбль (RF + CatBoost + GB)
        # =============================================
        logger.info("--- Ансамбль (RF + CatBoost + GB) ---")

        rf_w = max(rf_metrics['R2'], 0.01)
        cb_w = max(cb_metrics['R2'], 0.01)
        gb_w = max(gb_metrics['R2'], 0.01)
        total = rf_w + cb_w + gb_w

        ensemble_pred = (
            rf_w * rf_pred + cb_w * cb_pred + gb_w * gb_pred
        ) / total
        ens_metrics = self._evaluate(y_test, ensemble_pred, 'Ансамбль')
        all_metrics['Ансамбль'] = ens_metrics

        # =============================================
        # Итоги
        # =============================================
        best_name = max(all_metrics, key=lambda k: all_metrics[k]['R2'])
        self.best_model_name = best_name
        best_r2 = all_metrics[best_name]['R2']

        logger.info("=" * 60)
        logger.info(f"ЛУЧШАЯ МОДЕЛЬ: {best_name} (R²={best_r2:.4f})")
        logger.info("=" * 60)

        if best_r2 >= 0.7:
            logger.info("Требование ТЗ R² >= 0.7 — ВЫПОЛНЕНО")
        else:
            logger.warning(
                f"Требование ТЗ R² >= 0.7 — НЕ выполнено "
                f"(текущий: {best_r2:.4f})"
            )

        for name, m in all_metrics.items():
            if m['MAE%'] <= 20:
                logger.info(
                    f"  [{name}] MAE <= 20% — ВЫПОЛНЕНО ({m['MAE%']:.1f}%)"
                )
            else:
                logger.warning(
                    f"  [{name}] MAE <= 20% — НЕ выполнено ({m['MAE%']:.1f}%)"
                )

        return all_metrics

    def compare(self, X, y, top_skill_cols=None, test_size=0.2, random_state=42):
        return self.train(X, y, top_skill_cols, test_size, random_state)

    # -------------------------------------------------------
    # Предсказание на новых данных
    # -------------------------------------------------------
    def predict(self, X):
        if self.best_model_name is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")

        X_pred = X.copy()

        # Целевое кодирование (карты из train)
        if self.target_encoding_maps:
            for te_col, means in self.target_encoding_maps.items():
                cat_col = self.te_col_mapping.get(te_col)
                if cat_col and cat_col in X_pred.columns:
                    X_pred[te_col] = X_pred[cat_col].map(means).fillna(
                        self.te_fallback
                    )

        # CatBoost — cat_features + text_features
        cat_indices = self.models.get('_cat_indices', [])
        text_indices = self.models.get('_text_indices', [])
        pool = Pool(
            X_pred,
            cat_features=cat_indices,
            text_features=text_indices
        )
        cb_pred = self.models['catboost'].predict(pool)

        # RandomForest — только числовые
        rf_cols = self.models.get('_rf_columns', [])
        X_rf = self._prepare_rf_features(X_pred)
        for col in rf_cols:
            if col not in X_rf.columns:
                X_rf[col] = 0
        X_rf = X_rf[rf_cols]
        rf_pred = self.models['random_forest'].predict(X_rf)

        # GradientBoosting — те же признаки
        gb_pred = self.models['gradient_boosting'].predict(X_rf)

        if self.best_model_name == 'CatBoost':
            return cb_pred
        elif self.best_model_name == 'GradientBoosting':
            return gb_pred
        elif self.best_model_name == 'Ансамбль':
            return (cb_pred + rf_pred + gb_pred) / 3
        else:
            return rf_pred

    # -------------------------------------------------------
    # Сохранение / загрузка
    # -------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Модель сохранена: {path}")

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Модель загружена: {path}")
        return model