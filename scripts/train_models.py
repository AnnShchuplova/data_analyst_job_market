"""
Скрипт для обучения всех ML-моделей проекта DataTrack.

Обучает:
1. Модель прогнозирования зарплаты (RandomForest/CatBoost)
2. Модель кластеризации (KMeans)
3. Модель анализа временных рядов (ARIMA)

Использование:
    python scripts/train_models.py
    python scripts/train_models.py --data finaldata/month_dataset.csv --algorithm catboost
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Добавляем корень проекта в path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_processing.feature_engineering import FeatureEngineer
from src.ml.salary_predictor import SalaryPredictor
from src.ml.arima_analyzer import TimeSeriesAnalyzer


def train_salary_model(df: pd.DataFrame, algorithm: str = 'random_forest',
                       tune: bool = False) -> SalaryPredictor:
    """Обучение модели прогнозирования зарплаты."""
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ЗАРПЛАТЫ")
    print("="*60)

    fe = FeatureEngineer()
    df_prepared, feature_cols = fe.prepare_dataset_for_regression(df, fit=True)

    predictor = SalaryPredictor()
    metrics = predictor.train(
        df_prepared, feature_cols,
        algorithm=algorithm,
        tune_hyperparams=tune,
        remove_outliers=True,
        salary_range=(20000, 600000)
    )

    print(f"\nМетрики:")
    print(f"  MAE: {metrics['MAE']:,.0f} руб. ({metrics['MAE_percent']:.1f}% от средней ЗП)")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:,.0f} руб.")
    print(f"  Train size: {metrics['train_size']}, Test size: {metrics['test_size']}")
    print(f"  Features: {metrics['n_features']}")
    print(f"  Algorithm: {metrics['algorithm']}")

    if metrics['MAE_percent'] <= 20:
        print(f"  ✅ MAE в пределах 20% (ТЗ выполнено)")
    else:
        print(f"  ⚠️ MAE {metrics['MAE_percent']:.1f}% > 20% (ТЗ не выполнено)")

    if metrics['R2'] >= 0.7:
        print(f"  ✅ R2 >= 0.7 (ТЗ выполнено)")
    else:
        print(f"  ⚠️ R2 {metrics['R2']:.4f} < 0.7 (ТЗ не выполнено)")

    # Важность признаков
    importance = predictor.get_feature_importance()
    if importance is not None:
        print(f"\nТоп-15 признаков:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']:35s} {row['importance']:.4f}")

    # Сохраняем
    predictor.save()
    print(f"\nМодель сохранена: models/salary_model.joblib")
    print(f"Метрики сохранены: models/salary_metrics.json")

    return predictor

def train_arima_model(df: pd.DataFrame) -> TimeSeriesAnalyzer:
    """Обучение модели SARIMA (Seasonal ARIMA)."""
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ SARIMA (Seasonal ARIMA)")
    print("="*60)

    analyzer = TimeSeriesAnalyzer()

    # Подготовка временного ряда (обрезка до 5 апреля — далее были проблемы со сбором)
    ts = analyzer.prepare_time_series(df, freq='D', end_date='2026-04-05')

    if len(ts) < 5:
        print("Недостаточно данных для SARIMA (минимум 5 точек)!")
        return None

    # Проверка стационарности
    stationarity = analyzer.check_stationarity(ts)
    print(f"\nСтационарность: {'Да' if stationarity['is_stationary'] else 'Нет'}")
    print(f"  ADF statistic: {stationarity['adf_statistic']:.4f}")
    print(f"  p-value: {stationarity['p_value']:.4f}")

    # Обучение SARIMA (прогноз на 21 день — 3 недели)
    result = analyzer.train(ts, forecast_periods=21)

    metrics = result['metrics']
    print(f"\nМетрики:")
    print(f"  AIC: {metrics['AIC']:.2f}")
    print(f"  BIC: {metrics['BIC']:.2f}")
    print(f"  Исторический MAE: {metrics['MAE_historical']:.2f}")
    print(f"  Порядок (p,d,q): {metrics['order']}")
    print(f"  Сезонный порядок (P,D,Q,m): {metrics['seasonal_order']}")
    print(f"  Точек данных: {metrics['data_points']}")

    print(f"\nПрогноз на {len(result['forecast_mean'])} дней:")
    for date, val in result['forecast_mean'].items():
        conf = result['forecast_conf_int'].loc[date]
        print(f"  {date.strftime('%Y-%m-%d')}: {val:.0f} вакансий ({conf.iloc[0]:.0f} - {conf.iloc[1]:.0f})")

    analyzer.save()
    print(f"\nМодель сохранена: models/arima_model.joblib")
    print(f"Метрики сохранены: models/arima_metrics.json")

    return analyzer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Обучение ML-моделей DataTrack')
    parser.add_argument('--data', type=str, default=None, help='Путь к CSV файлу')
    parser.add_argument('--algorithm', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'catboost'],
                        help='Алгоритм для модели зарплаты')
    parser.add_argument('--tune', action='store_true', help='Подбор гиперпараметров')
    parser.add_argument('--skip-salary', action='store_true', help='Пропустить обучение модели зарплаты')
    parser.add_argument('--skip-arima', action='store_true', help='Пропустить ARIMA')
    args = parser.parse_args()

    # Определяем путь к данным
    if args.data:
        data_path = args.data
    else:
        processed_dir = os.path.join(project_root, "finaldata")
        if not os.path.exists(processed_dir):
            print(f"Папка {processed_dir} не найдена!")
            sys.exit(1)
        csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not csv_files:
            print("CSV файлы не найдены в finaldata/")
            sys.exit(1)
        latest = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
        data_path = os.path.join(processed_dir, latest)

    print(f"Данные: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"Загружено {len(df)} записей")

    # Обучение моделей
    if not args.skip_salary:
        train_salary_model(df, algorithm=args.algorithm, tune=args.tune)

    if not args.skip_arima:
        train_arima_model(df)

    print(f"\nМодели сохранены в папке: {os.path.join(project_root, 'models')}")
    print("Для запуска приложения: streamlit run app/main.py")


if __name__ == "__main__":
    main()
