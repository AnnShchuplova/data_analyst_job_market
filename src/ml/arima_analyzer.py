"""
Модуль анализа временных рядов (SARIMA).

Согласно ЧТЗ раздел «Модель анализа временных рядов»:
- Цель: Прогнозирование количества публикуемых вакансий на 3 месяца вперёд
- Данные: временной ряд по датам публикации (агрегация по дням/неделям)
- Проверка ряда на стационарность (тест Дики-Фуллера)
- Подбор параметров (p,d,q) и сезонных (P,D,Q,m) модели SARIMA
- Визуализация (график с историческими данными и прогнозом)
- Сохранение модели (arima_model.joblib)

Используется SARIMA (Seasonal ARIMA) для учёта недельной/годичной цикличности.
Лог-трансформация (log1p/expm1) для корректной работы с count-данными.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

# Маппинг freq → сезонный период m
FREQ_TO_SEASONAL_PERIOD = {
    'D': 7,    # дневные данные → недельная сезонность
    'W': 52,   # недельные данные → годичная сезонность
    'M': 12,   # месячные данные → годичная сезонность
}


class TimeSeriesAnalyzer:
    """Модель SARIMA для анализа и прогнозирования временных рядов вакансий.
    
    В отличие от обычного ARIMA, SARIMA учитывает сезонные (циклические) 
    паттерны через параметр seasonal_order=(P, D, Q, m), где:
    - P — сезонный авторегрессионный порядок
    - D — сезонный порядок дифференцирования
    - Q — сезонный порядок скользящего среднего
    - m — период сезонности (7 для дневных, 52 для недельных данных)
    """
    
    def __init__(self):
        self.model = None
        self.model_params = {}
        self.metrics = {}
        self._is_trained = False
        self._stationarity_result = None
        self._seasonal_period = 7  # по умолчанию — недельная (для дневных данных)
        self._log_transform = True  # использовать log1p для count-данных
    
    def prepare_time_series(self, df: pd.DataFrame, 
                            date_column: str = 'published_at',
                            freq: str = 'D',
                            end_date: str = None) -> pd.Series:
        """Подготовка временного ряда из данных о вакансиях.
        
        Агрегирует количество вакансий по дням/неделям/месяцам.
        
        Parameters:
        -----------
        df : DataFrame с данными о вакансиях
        date_column : колонка с датой публикации
        freq : частота агрегации ('D' - день, 'W' - неделя, 'M' - месяц)
        end_date : обрезать данные по этой дате (формат 'YYYY-MM-DD'), включительно
        
        Returns:
        --------
        pd.Series : временной ряд (индекс - дата, значение - количество вакансий)
        """
        df = df.copy()
        
        if date_column not in df.columns:
            raise ValueError(f"Колонка '{date_column}' не найдена")
        
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.dropna(subset=[date_column])
        
        # Обрезка по дате (если указана)
        if end_date is not None:
            cutoff = pd.to_datetime(end_date)
            # Если колонка с таймзоной — убираем таймзону для сравнения
            if df[date_column].dt.tz is not None:
                df[date_column] = df[date_column].dt.tz_localize(None)
            before = len(df)
            df = df[df[date_column] <= cutoff]
            removed = before - len(df)
            if removed > 0:
                logger.info(f"Обрезка по дате {end_date}: удалено {removed} записей, "
                           f"осталось {len(df)}")
        
        df = df.set_index(date_column)
        
        # Агрегация по частоте
        ts = df.resample(freq).size()
        ts.name = 'vacancies_count'
        
        # Заполнение пропущенных значений нулями
        ts = ts.fillna(0).astype(int)
        
        # Устанавливаем сезонный период
        self._seasonal_period = FREQ_TO_SEASONAL_PERIOD.get(freq, 7)
        
        logger.info(f"Временной ряд: {len(ts)} точек, частота '{freq}', сезонный период m={self._seasonal_period}")
        logger.info(f"Диапазон: {ts.index.min()} - {ts.index.max()}")
        logger.info(f"Среднее кол-во вакансий за период: {ts.mean():.1f}")
        
        return ts
    
    def check_stationarity(self, ts: pd.Series, significance: float = 0.05) -> dict:
        """Проверка ряда на стационарность тестом Дики-Фуллера."""
        logger.info("Проверка стационарности временного ряда (ADF-тест)...")
        
        result = adfuller(ts, autolag='AIC')
        
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        is_stationary = p_value < significance
        
        self._stationarity_result = {
            'adf_statistic': round(adf_stat, 4),
            'p_value': round(p_value, 4),
            'is_stationary': is_stationary,
            'significance_level': significance,
            'critical_values': {k: round(v, 4) for k, v in critical_values.items()}
        }
        
        logger.info(f"  ADF статистика: {adf_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Стационарный: {'Да' if is_stationary else 'Нет'}")
        
        if not is_stationary:
            logger.info("  Ряд нестационарен, необходимо дифференцирование (d>=1)")
        
        return self._stationarity_result
    
    def _determine_arima_order(self, ts: pd.Series) -> tuple:
        """Определение параметров ARIMA (p,d,q) на основе ADF-теста."""
        d = 0
        test_ts = ts.copy()
        
        if self._stationarity_result and not self._stationarity_result['is_stationary']:
            test_ts = ts.diff().dropna()
            result = adfuller(test_ts, autolag='AIC')
            if result[1] < 0.05:
                d = 1
                logger.info("  Первое дифференцирование достаточно (d=1)")
            else:
                test_ts = ts.diff().diff().dropna()
                result = adfuller(test_ts, autolag='AIC')
                if result[1] < 0.05:
                    d = 2
                    logger.info("  Необходимо второе дифференцирование (d=2)")
                else:
                    d = 1
                    logger.info("  Используем d=1 (ряд сложный)")
        
        # p и q — по длине ряда, но ограничиваем для стабильности
        n = len(ts)
        if n < 30:
            p, q = 1, 1
        elif n < 60:
            p, q = 2, 1
        else:
            p, q = 2, 2
        
        logger.info(f"  Определённые параметры ARIMA: (p={p}, d={d}, q={q})")
        return (p, d, q)
    
    def _determine_seasonal_order(self, ts: pd.Series, m: int) -> tuple:
        """Определение сезонных параметров (P, D, Q, m).
        
        Сезонное дифференцирование D проверяется через seasonal ADF-test.
        Типичные начальные значения: P=1, D=1, Q=1.
        """
        D = 0
        P, Q = 1, 1
        
        # Проверяем нужна ли сезонная дифференциация
        if len(ts) >= 2 * m:
            seasonal_diff = ts.diff(m).dropna()
            if len(seasonal_diff) >= 5:
                try:
                    result = adfuller(seasonal_diff, autolag='AIC')
                    if result[1] < 0.05:
                        D = 1
                        logger.info(f"  Сезонное дифференцирование эффективно (D=1, m={m})")
                    else:
                        D = 0
                        logger.info(f"  Сезонное дифференцирование не требуется (D=0)")
                except Exception as e:
                    logger.info(f"  Не удалось проверить сезонную стационарность: {e}")
                    D = 1  # безопасное значение по умолчанию
        else:
            # Данных меньше 2 сезонных циклов — используем D=0
            logger.info(f"  Данных недостаточно для сезонного анализа (< 2*m={2*m}), D=0")
            D = 0
        
        seasonal_order = (P, D, Q, m)
        logger.info(f"  Сезонные параметры: {seasonal_order}")
        return seasonal_order
    
    def _inverse_log(self, values: np.ndarray) -> np.ndarray:
        """Обратная лог-трансформация: expm1(x) = exp(x) - 1."""
        return np.expm1(values)
    
    def _simulate_confidence_intervals(self, steps: int, n_simulations: int = 1000,
                                        alpha: float = 0.05) -> tuple:
        """Вычисление доверительных интервалов через симуляцию Монте-Карло.
        
        Вместо naively применяя expm1 к границам из лог-пространства (что раздувает
        верхнюю границу экспоненциально), мы:
        1. Генерируем n_simulations симулированных путей прогноза в лог-пространстве
        2. Конвертируем каждый путь в исходные единицы через expm1
        3. Берём перцентили (alpha/2 и 1-alpha/2) от каждой точки
        
        Это даёт корректные, не раздутые доверительные интервалы.
        """
        try:
            simulations = self.model.simulate(
                nsimulations=n_simulations,
                steps=steps
            )
            # simulations shape: (n_simulations, steps)
            # Конвертируем из лог-пространства в исходное
            simulations_original = np.expm1(simulations.values)
            
            # Перцентили
            lower = np.percentile(simulations_original, 100 * alpha / 2, axis=0)
            upper = np.percentile(simulations_original, 100 * (1 - alpha / 2), axis=0)
            
            logger.info(f"  Симуляция: {n_simulations} путей, alpha={alpha}")
            logger.info(f"  Пример CI (шаг 0): [{lower[0]:.0f}, {upper[0]:.0f}]")
            
            return np.maximum(0, lower), upper
        except Exception as e:
            logger.warning(f"Симуляция не удалась ({e}), падбэк на обычный CI")
            forecast = self.model.get_forecast(steps=steps)
            ci = forecast.conf_int()
            return (self._inverse_log(ci.iloc[:, 0].values),
                    self._inverse_log(ci.iloc[:, 1].values))
    
    def train(self, ts: pd.Series, 
              order: tuple = None,
              seasonal_order: tuple = None,
              forecast_periods: int = 21) -> dict:
        """Обучение модели SARIMA и построение прогноза.
        
        Parameters:
        -----------
        ts : временной ряд (pd.Series)
        order : параметры ARIMA (p, d, q). Если None, определяются автоматически
        seasonal_order : сезонные параметры (P, D, Q, m). Если None, определяются автоматически
        forecast_periods : количество периодов для прогноза
        
        Returns:
        --------
        dict : прогноз и метрики
        """
        logger.info("Обучение модели SARIMA")
        
        m = self._seasonal_period
        
        # Лог-трансформация
        if self._log_transform:
            ts_for_model = np.log1p(ts)
            logger.info("Применена лог-трансформация log1p к временному ряду")
        else:
            ts_for_model = ts
        
        # Проверка стационарности
        self.check_stationarity(ts)
        
        # Автоподбор параметров
        if order is None:
            order = self._determine_arima_order(ts_for_model)
        
        if seasonal_order is None:
            seasonal_order = self._determine_seasonal_order(ts_for_model, m)
        
        self.model_params = {
            'order': order,
            'seasonal_order': seasonal_order,
            'forecast_periods': forecast_periods,
            'freq': ts.index.freqstr if ts.index.freq else 'D',
            'log_transform': self._log_transform
        }
        
        logger.info(f"Параметры SARIMA: order={order}, seasonal_order={seasonal_order}")
        logger.info(f"Периоды прогноза: {forecast_periods}")
        
        try:
            self.model = SARIMAX(
                ts_for_model, 
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = self.model.fit(disp=False, maxiter=200)
            self._is_trained = True
            
            logger.info(f"SARIMA {order} x {seasonal_order} обучена успешно")
            logger.info(f"  AIC: {self.model.aic:.2f}")
            logger.info(f"  BIC: {self.model.bic:.2f}")
            
        except Exception as e:
            logger.warning(f"Ошибка обучения SARIMA {order} x {seasonal_order}: {e}")
            logger.info("Пробуем упрощённые параметры...")
            
            # Падбэк 1: уменьшаем сезонные параметры
            try:
                simple_seasonal = (1, 0, 1, m)
                logger.info(f"  Падбэк: seasonal_order={simple_seasonal}")
                self.model = SARIMAX(
                    ts_for_model, 
                    order=order,
                    seasonal_order=simple_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.model = self.model.fit(disp=False, maxiter=200)
                self._is_trained = True
                seasonal_order = simple_seasonal
                self.model_params['seasonal_order'] = seasonal_order
                logger.info(f"SARIMA обучена (резервная сезонная модель)")
            except Exception as e2:
                # Падбэк 2: без сезонности вообще
                logger.warning(f"Падбэк 1 тоже упал: {e2}")
                logger.info("  Падбэк 2: обычный ARIMA без сезонности")
                order = (1, 0, 1)
                seasonal_order = (0, 0, 0, 0)
                self.model_params['order'] = order
                self.model_params['seasonal_order'] = seasonal_order
                self.model = SARIMAX(ts_for_model, order=order, enforce_stationarity=False)
                self.model = self.model.fit(disp=False)
                self._is_trained = True
                logger.info(f"ARIMA {order} обучена (без сезонности)")
        
        # Прогноз — через симуляцию для корректных доверительных интервалов
        # (expm1 на границах лог-пространства раздувает верхнюю границу экспоненциально)
        forecast_log = self.model.get_forecast(steps=forecast_periods)
        forecast_mean_log = forecast_log.predicted_mean

        if self._log_transform:
            forecast_mean = self._inverse_log(forecast_mean_log.values)
            fitted_values = self._inverse_log(self.model.fittedvalues.values)
            fitted_values = pd.Series(fitted_values, index=self.model.fittedvalues.index)

            # Симуляция Монте-Карло: генерируем 1000 путей в лог-пространстве,
            # конвертируем в исходное через expm1, берём перцентили
            lower_bound, upper_bound = self._simulate_confidence_intervals(
                forecast_periods, n_simulations=1000, alpha=0.05
            )
            logger.info(f"  Доверительный интервал через симуляцию (1000 путей)")
        else:
            forecast_mean = forecast_mean_log.values
            forecast_conf_int_log = forecast_log.conf_int()
            lower_bound = forecast_conf_int_log.iloc[:, 0].values
            upper_bound = forecast_conf_int_log.iloc[:, 1].values
            fitted_values = self.model.fittedvalues
        
        # Убираем мелкие отрицательные из-за погрешности float
        forecast_mean = np.maximum(0, forecast_mean)
        lower_bound = np.maximum(0, lower_bound)
        fitted_values = np.maximum(0, fitted_values)
        
        # Метрики
        actual_aligned = ts.loc[self.model.fittedvalues.index]
        residuals = actual_aligned - fitted_values
        mae = np.mean(np.abs(residuals))
        
        self.metrics = {
            'AIC': round(self.model.aic, 2),
            'BIC': round(self.model.bic, 2),
            'MAE_historical': round(mae, 2),
            'order': list(order),
            'seasonal_order': list(seasonal_order),
            'forecast_periods': forecast_periods,
            'data_points': len(ts),
            'seasonal_period': m,
            'log_transform': self._log_transform,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Метрики:")
        logger.info(f"  AIC: {self.model.aic:.2f}")
        logger.info(f"  Исторический MAE: {mae:.2f}")
        
        # Формируем Series с правильным индексом
        forecast_mean_s = pd.Series(forecast_mean, index=forecast_mean_log.index)
        lower_bound_s = pd.Series(lower_bound, index=forecast_mean_log.index)
        upper_bound_s = pd.Series(upper_bound, index=forecast_mean_log.index)
        
        result = {
            'historical': ts,
            'fitted': fitted_values,
            'forecast_mean': forecast_mean_s,
            'forecast_conf_int': pd.concat([lower_bound_s, upper_bound_s], axis=1),
            'metrics': self.metrics,
            'order': order,
            'seasonal_order': seasonal_order,
            'stationarity': self._stationarity_result
        }
        
        return result
    
    def forecast(self, periods: int = 21) -> pd.DataFrame:
        """Получение прогноза от обученной модели."""
        if not self._is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        forecast = self.model.get_forecast(steps=periods)
        forecast_mean = forecast.predicted_mean
        
        if self._log_transform:
            forecast_values = self._inverse_log(forecast_mean.values)
            # Доверительные интервалы через симуляцию
            lower_values, upper_values = self._simulate_confidence_intervals(
                periods, n_simulations=1000, alpha=0.05
            )
            forecast_values = np.maximum(0, np.round(forecast_values))
            lower_values = np.maximum(0, np.round(lower_values))
            upper_values = np.round(upper_values)
        else:
            forecast_conf_int = forecast.conf_int()
            forecast_values = np.maximum(0, np.round(forecast_mean.values))
            lower_values = np.maximum(0, np.round(forecast_conf_int.iloc[:, 0].values))
            upper_values = np.round(forecast_conf_int.iloc[:, 1].values)
        
        result_df = pd.DataFrame({
            'date': forecast_mean.index,
            'forecast': forecast_values,
            'lower_bound': lower_values,
            'upper_bound': upper_values
        })
        
        return result_df
    
    def save(self, filepath: str = None):
        """Сохранение обученной модели (arima_model.joblib)."""
        import joblib
        
        if not self._is_trained:
            raise ValueError("Модель не обучена, нечего сохранять.")
        
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, "arima_model.joblib")
        
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'metrics': self.metrics,
            'stationarity': self._stationarity_result,
            'log_transform': self._log_transform,
            'seasonal_period': self._seasonal_period,
            'is_trained': self._is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Модель SARIMA сохранена: {filepath}")
        
        # Сохранение метрик
        metrics_path = os.path.join(os.path.dirname(filepath), "arima_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str = None):
        """Загрузка обученной модели."""
        import joblib
        
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "arima_model.joblib")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Модель не найдена: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_params = model_data['model_params']
        self.metrics = model_data['metrics']
        self._stationarity_result = model_data.get('stationarity')
        self._log_transform = model_data.get('log_transform', True)
        self._seasonal_period = model_data.get('seasonal_period', 7)
        self._is_trained = model_data['is_trained']
        
        logger.info(f"Модель SARIMA загружена: {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "finaldata", "month_dataset_20260420_205719.csv"
    )
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        analyzer = TimeSeriesAnalyzer()
        ts = analyzer.prepare_time_series(df, freq='D', end_date='2026-04-05')
        
        result = analyzer.train(ts, forecast_periods=21)
        
        print(f"\nМетрики SARIMA: {result['metrics']}")
        print(f"\nПрогноз на {len(result['forecast_mean'])} дней:")
        for date, val in result['forecast_mean'].items():
            print(f"  {date.strftime('%Y-%m-%d')}: {val:.0f} вакансий")
        
        analyzer.save()
    else:
        print(f"Файл не найден: {data_path}")
