"""
Тесты ML-модулей: SalaryPredictor и TimeSeriesAnalyzer.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest


class TestSalaryPredictor:
    """Тесты для src.ml.salary_predictor.SalaryPredictor."""

    def test_init_default_attributes(self):
        """Проверка, что после инициализации все атрибуты установлены."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()

        assert sp.models == {}
        assert sp.best_model_name is None
        assert sp.feature_engineer is None
        assert sp.ensemble_weights == {}
        assert sp.best_mae is None
        assert sp.best_r2 is None

    def test_cat_and_text_columns_defined(self):
        """Проверка, что cat_columns и text_columns содержат ожидаемые имена."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()

        assert "experience_name" in sp.cat_columns
        assert "area_name" in sp.cat_columns
        assert "main_role_name" in sp.cat_columns
        assert "requirement" in sp.text_columns
        assert "responsibility" in sp.text_columns

    def test_add_target_encoding_creates_columns(self, salary_regression_data):
        """Целевое кодирование должно создавать новые колонки и заполнять fallback."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        X_train = X.iloc[:150].copy()
        X_test = X.iloc[150:].copy()
        y_train = y.iloc[:150]

        X_train_enc, X_test_enc = sp._add_target_encoding(X_train, X_test, y_train)

        # Проверяем, что TE-колонки созданы
        assert "role_mean_salary" in X_train_enc.columns
        assert "region_mean_salary" in X_train_enc.columns

        # Fallback заполнен
        assert sp.te_fallback is not None
        assert sp.te_fallback > 0

        # Нет NaN после заполнения
        assert not X_train_enc["role_mean_salary"].isna().any()
        assert not X_test_enc["role_mean_salary"].isna().any()

    def test_add_target_encoding_no_data_leakage(self, salary_regression_data):
        """TE-значения для теста не должны зависеть от y_test."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        X_train = X.iloc[:150].copy()
        X_test = X.iloc[150:].copy()
        y_train = y.iloc[:150]

        # Два вызова с разным y_train должны давать разные TE-карты
        _, X_test_enc1 = sp._add_target_encoding(X_train.copy(), X_test.copy(), y_train)

        sp2 = SalaryPredictor()
        y_train_shifted = y_train * 2
        _, X_test_enc2 = sp2._add_target_encoding(X_train.copy(), X_test.copy(), y_train_shifted)

        # Средние значения должны отличаться (масштаб y отличается в 2 раза)
        assert not np.allclose(
            X_test_enc1["role_mean_salary"].mean(),
            X_test_enc2["role_mean_salary"].mean(),
            atol=1000,
        )

    def test_prepare_rf_features_drops_cat_and_text(self, salary_regression_data):
        """_prepare_rf_features должен удалять категориальные и текстовые колонки."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        X_rf = sp._prepare_rf_features(X)

        for col in sp.cat_columns:
            assert col not in X_rf.columns
        for col in sp.text_columns:
            assert col not in X_rf.columns

        # Числовые колонки остаются
        assert "skill_count" in X_rf.columns

    def test_get_cat_feature_indices(self, salary_regression_data):
        """Индексы cat-признаков должны корректно определяться."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data

        indices = sp._get_cat_feature_indices(X)

        assert len(indices) == len(sp.cat_columns)
        for idx in indices:
            assert isinstance(idx, int)
            assert 0 <= idx < len(X.columns)

    def test_get_text_feature_indices(self, salary_regression_data):
        """Индексы text-признаков должны корректно определяться."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data

        indices = sp._get_text_feature_indices(X)

        assert len(indices) == len(sp.text_columns)

    def test_evaluate_returns_all_metrics(self):
        """_evaluate должен возвращать словарь с MAE, MAE%, RMSE, R2, MAPE."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        y_true = np.array([100, 200, 300], dtype=float)
        y_pred = np.array([110, 190, 310], dtype=float)

        metrics = sp._evaluate(y_true, y_pred, "test_model")

        assert "MAE" in metrics
        assert "MAE%" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert "MAPE" in metrics
        assert metrics["MAE"] > 0
        assert metrics["R2"] > 0

    def test_evaluate_perfect_predictions(self):
        """При идеальных предсказаниях R2 должен быть ~1.0, MAE ~0."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        y = np.array([100, 200, 300], dtype=float)
        metrics = sp._evaluate(y, y, "perfect")

        assert metrics["R2"] > 0.999
        assert metrics["MAE"] < 1.0

    def test_predict_raises_when_not_trained(self, salary_regression_data):
        """predict() должен выбрасывать ValueError до обучения."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"]).head(5)

        with pytest.raises(ValueError, match="не обучена"):
            sp.predict(X)

    def test_train_returns_all_model_metrics(self, salary_regression_data):
        """train() должен возвращать метрики для всех 4 моделей."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        metrics = sp.train(X, y, test_size=0.2, random_state=42)

        assert "RandomForest" in metrics
        assert "CatBoost" in metrics
        assert "GradientBoosting" in metrics
        assert "Ансамбль" in metrics

        for model_metrics in metrics.values():
            assert "MAE" in model_metrics
            assert "R2" in model_metrics

    def test_train_sets_best_model_name(self, salary_regression_data):
        """После train() best_model_name должен быть установлен."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        sp.train(X, y, test_size=0.2, random_state=42)

        assert sp.best_model_name is not None
        assert sp.best_model_name in [
            "RandomForest", "CatBoost", "GradientBoosting", "Ансамбль"
        ]

    def test_train_sets_ensemble_weights(self, salary_regression_data):
        """После train() ensemble_weights должен содержать ключи для всех моделей."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        sp.train(X, y, test_size=0.2, random_state=42)

        assert "RandomForest" in sp.ensemble_weights
        assert "CatBoost" in sp.ensemble_weights
        assert "GradientBoosting" in sp.ensemble_weights

        # Веса суммируются в 1.0
        total_weight = sum(sp.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_train_stores_models(self, salary_regression_data):
        """После train() модели должны быть доступны в self.models."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        sp.train(X, y, test_size=0.2, random_state=42)

        assert "random_forest" in sp.models
        assert "catboost" in sp.models
        assert "gradient_boosting" in sp.models

    def test_predict_returns_array(self, salary_regression_data):
        """predict() должен возвращать массив предсказаний."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        sp.train(X, y, test_size=0.2, random_state=42)

        X_new = X.head(10)
        predictions = sp.predict(X_new)

        assert len(predictions) == 10
        assert all(p > 0 for p in predictions)

    def test_predict_ensemble_weights_applied(self, salary_regression_data):
        """Если best_model == 'Ансамбль', predict() использует веса."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        metrics = sp.train(X, y, test_size=0.2, random_state=42)
        sp.best_model_name = "Ансамбль"

        X_new = X.head(5)
        predictions = sp.predict(X_new)

        assert len(predictions) == 5
        assert all(p > 0 for p in predictions)

    def test_save_and_load(self, salary_regression_data, tmp_path):
        """save() / load() должны корректно сохранять и загружать модель."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        sp.train(X, y, test_size=0.2, random_state=42)

        model_path = str(tmp_path / "salary_model.pkl")
        sp.save(model_path)

        assert os.path.exists(model_path)

        loaded = SalaryPredictor.load(model_path)
        assert loaded.best_model_name == sp.best_model_name
        assert loaded.best_r2 == sp.best_r2
        assert "random_forest" in loaded.models

        X_new = X.head(5)
        pred_orig = sp.predict(X_new)
        pred_loaded = loaded.predict(X_new)
        np.testing.assert_array_almost_equal(pred_orig, pred_loaded, decimal=0)

    def test_load_compatibility_missing_fields(self, tmp_path):
        """load() должен восстанавливать старые модели без новых полей."""
        import pickle
        from src.ml.salary_predictor import SalaryPredictor

        # Создаем "старую" модель без новых полей
        sp = SalaryPredictor()
        # Удаляем новые поля
        del sp.te_col_mapping
        del sp.best_mae
        del sp.best_mae_pct
        del sp.best_r2
        del sp.ensemble_weights

        path = str(tmp_path / "old_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(sp, f)

        loaded = SalaryPredictor.load(path)

        assert hasattr(loaded, "te_col_mapping")
        assert hasattr(loaded, "best_mae")
        assert hasattr(loaded, "best_mae_pct")
        assert hasattr(loaded, "best_r2")
        assert hasattr(loaded, "ensemble_weights")

    def test_compare_alias(self, salary_regression_data):
        """compare() — это алиас для train(), должен возвращать то же самое."""
        from src.ml.salary_predictor import SalaryPredictor

        sp = SalaryPredictor()
        X = salary_regression_data.drop(columns=["target_salary"])
        y = salary_regression_data["target_salary"]

        result = sp.compare(X, y, test_size=0.2, random_state=42)

        assert "RandomForest" in result


# ==================== TimeSeriesAnalyzer ====================

class TestTimeSeriesAnalyzer:
    """Тесты для src.ml.arima_analyzer.TimeSeriesAnalyzer."""

    def test_init_defaults(self):
        """Проверка начальных значений атрибутов."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()

        assert analyzer.model is None
        assert analyzer._is_trained is False
        assert analyzer._seasonal_period == 7
        assert analyzer._log_transform is True

    def test_prepare_time_series_basic(self, time_series_daily):
        """Базовая подготовка временного ряда из DataFrame."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        df = pd.DataFrame({
            "published_at": time_series_daily.index,
            "vacancy_id": range(len(time_series_daily)),
        })

        ts = analyzer.prepare_time_series(df, date_column="published_at", freq="D")

        assert isinstance(ts, pd.Series)
        assert len(ts) == 90
        assert ts.name == "vacancies_count"
        assert analyzer._seasonal_period == 7

    def test_prepare_time_series_missing_column_raises(self):
        """Должен выбрасывать ValueError при отсутствии колонки с датой."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        df = pd.DataFrame({"some_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="не найдена"):
            analyzer.prepare_time_series(df, date_column="nonexistent")

    def test_prepare_time_series_weekly_freq(self, time_series_daily):
        """При freq='W' сезонный период должен быть 52."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        df = pd.DataFrame({
            "published_at": time_series_daily.index,
        })

        ts = analyzer.prepare_time_series(df, date_column="published_at", freq="W")

        assert analyzer._seasonal_period == 52

    def test_prepare_time_series_monthly_freq(self, time_series_daily):
        """При freq='M' сезонный период должен быть 12."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        df = pd.DataFrame({
            "published_at": time_series_daily.index,
        })

        ts = analyzer.prepare_time_series(df, date_column="published_at", freq="ME")

        assert analyzer._seasonal_period == 12

    def test_prepare_time_series_with_end_date(self):
        """Обрезка по end_date должна уменьшить количество записей."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        dates = pd.date_range("2025-10-01", periods=60, freq="D")
        df = pd.DataFrame({"published_at": dates})

        ts = analyzer.prepare_time_series(
            df, date_column="published_at", freq="D", end_date="2025-10-30"
        )

        # Последняя дата не должна превышать 2025-10-30
        assert ts.index.max() <= pd.Timestamp("2025-10-30")

    def test_check_stationarity_returns_dict(self, time_series_daily):
        """check_stationarity должен возвращать словарь с нужными ключами."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        result = analyzer.check_stationarity(time_series_daily)

        assert isinstance(result, dict)
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result
        assert "significance_level" in result
        assert "critical_values" in result

    def test_check_stationarity_stores_result(self, time_series_daily):
        """Результат должен сохраняться в _stationarity_result."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.check_stationarity(time_series_daily)

        assert analyzer._stationarity_result is not None
        assert "p_value" in analyzer._stationarity_result

    def test_determine_arima_order(self, time_series_daily):
        """_determine_arima_order должен возвращать кортеж (p, d, q)."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.check_stationarity(time_series_daily)

        order = analyzer._determine_arima_order(time_series_daily)

        assert isinstance(order, tuple)
        assert len(order) == 3
        p, d, q = order
        assert p >= 1
        assert d >= 0
        assert q >= 1

    def test_determine_arima_order_short_series(self):
        """Для короткого ряда (< 30 точек) p=1, q=1."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        short_ts = pd.Series(np.random.randint(1, 20, 20))

        order = analyzer._determine_arima_order(short_ts)

        assert order == (1, 0, 1)

    def test_determine_seasonal_order(self, time_series_daily):
        """_determine_seasonal_order должен возвращать (P, D, Q, m)."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        m = 7

        seasonal = analyzer._determine_seasonal_order(time_series_daily, m)

        assert isinstance(seasonal, tuple)
        assert len(seasonal) == 4
        P, D, Q, m_out = seasonal
        assert P >= 1
        assert D >= 0
        assert Q >= 1
        assert m_out == m

    def test_inverse_log(self):
        """_inverse_log должен корректно применять expm1."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()

        result = analyzer._inverse_log(np.array([0.0, 0.693, 1.099]))
        expected = np.expm1(np.array([0.0, 0.693, 1.099]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_train_returns_result_dict(self, time_series_daily):
        """train() должен возвращать словарь с ключами historical, forecast и др."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        result = analyzer.train(time_series_daily, forecast_periods=7)

        assert "historical" in result
        assert "fitted" in result
        assert "forecast_mean" in result
        assert "forecast_conf_int" in result
        assert "metrics" in result
        assert "order" in result
        assert "seasonal_order" in result
        assert "stationarity" in result

    def test_train_sets_is_trained(self, time_series_daily):
        """После train() _is_trained должен быть True."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.train(time_series_daily, forecast_periods=7)

        assert analyzer._is_trained is True
        assert analyzer.model is not None

    def test_train_metrics_contain_aic(self, time_series_daily):
        """Метрики после обучения должны содержать AIC и BIC."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        result = analyzer.train(time_series_daily, forecast_periods=7)

        assert "AIC" in result["metrics"]
        assert "BIC" in result["metrics"]
        # AIC может быть отрицательным для небольших рядов — это корректно
        assert isinstance(result["metrics"]["AIC"], float)

    def test_train_forecast_mean_non_negative(self, time_series_daily):
        """Прогнозные значения должны быть >= 0."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        result = analyzer.train(time_series_daily, forecast_periods=7)

        assert all(result["forecast_mean"] >= 0)

    def test_train_with_explicit_order(self, time_series_daily):
        """train() с явными параметрами order / seasonal_order."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        result = analyzer.train(
            time_series_daily,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            forecast_periods=7,
        )

        assert result["order"] == (1, 0, 1)
        assert result["seasonal_order"] == (1, 0, 1, 7)

    def test_forecast_raises_when_not_trained(self):
        """forecast() должен выбрасывать ValueError до обучения."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()

        with pytest.raises(ValueError, match="не обучена"):
            analyzer.forecast(periods=7)

    def test_forecast_returns_dataframe(self, time_series_daily):
        """forecast() должен возвращать DataFrame с колонками date, forecast, lower, upper."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.train(time_series_daily, forecast_periods=7)
        result = analyzer.forecast(periods=7)

        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "forecast" in result.columns
        assert "lower_bound" in result.columns
        assert "upper_bound" in result.columns
        assert len(result) == 7

    def test_forecast_upper_ge_lower(self, time_series_daily):
        """В доверительных интервалах upper >= lower."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.train(time_series_daily, forecast_periods=7)
        result = analyzer.forecast(periods=7)

        assert all(result["upper_bound"] >= result["lower_bound"])

    def test_save_and_load(self, time_series_daily, tmp_path):
        """save() / load() должны корректно сохранять и загружать модель."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.train(time_series_daily, forecast_periods=7)

        model_path = str(tmp_path / "arima_model.joblib")
        analyzer.save(model_path)

        assert os.path.exists(model_path)

        analyzer2 = TimeSeriesAnalyzer()
        analyzer2.load(model_path)

        assert analyzer2._is_trained is True
        assert analyzer2.model is not None
        assert analyzer2.metrics["AIC"] == analyzer.metrics["AIC"]

    def test_save_raises_when_not_trained(self, tmp_path):
        """save() должен выбрасывать ValueError до обучения."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()

        with pytest.raises(ValueError, match="не обучена"):
            analyzer.save(str(tmp_path / "model.joblib"))

    def test_load_raises_when_file_not_found(self, tmp_path):
        """load() должен выбрасывать FileNotFoundError при отсутствии файла."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()

        with pytest.raises(FileNotFoundError, match="не найдена"):
            analyzer.load(str(tmp_path / "nonexistent.joblib"))

    def test_simulate_confidence_intervals_fallback(self, time_series_daily):
        """При ошибке симуляции должен срабатывать fallback на обычный CI."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer.train(time_series_daily, forecast_periods=5)

        original_simulate = analyzer.model.simulate

        def failing_simulate(*args, **kwargs):
            raise RuntimeError("Simulate error")

        analyzer.model.simulate = failing_simulate

        # сработает fallback
        lower, upper = analyzer._simulate_confidence_intervals(5)

        assert len(lower) == 5
        assert len(upper) == 5
        assert all(upper >= lower)

    def test_train_no_log_transform(self, time_series_daily):
        """Обучение без лог-трансформации."""
        from src.ml.arima_analyzer import TimeSeriesAnalyzer

        analyzer = TimeSeriesAnalyzer()
        analyzer._log_transform = False

        result = analyzer.train(time_series_daily, forecast_periods=7)

        assert result["metrics"]["log_transform"] is False
        assert all(result["forecast_mean"] >= 0)
