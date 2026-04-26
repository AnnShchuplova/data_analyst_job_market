"""
Скрипт обучения ML-моделей предсказания зарплаты.

Использование:
    python scripts/train_models.py                          # Обучение лучшей модели
    python scripts/train_models.py --compare                # Сравнение всех моделей
    python scripts/train_models.py --compare --skip-arima   # Сравнение без ARIMA

Данные: ищется последний CSV файл в директории finaldata/
Целевая переменная: salary_avg
Навыки: skills_list
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Добавляем корень проекта в путь для импортов
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.feature_engineering import FeatureEngineer
from src.ml.salary_predictor import SalaryPredictor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_data_file():
    """Поиск файла данных в директории finaldata/."""
    finaldata_dir = PROJECT_ROOT / 'finaldata'

    if not finaldata_dir.exists():
        raise FileNotFoundError(
            f"Директория с данными не найдена: {finaldata_dir}"
        )

    # Ищем CSV файлы с датасетом
    csv_files = sorted(finaldata_dir.glob('month_dataset_*.csv'))

    if not csv_files:
        # Пробуем другие паттерны
        csv_files = sorted(finaldata_dir.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(
            f"CSV файлы не найдены в {finaldata_dir}"
        )

    # Берём самый свежий по имени (обычно содержит дату)
    latest = csv_files[-1]
    logger.info(f"Найден файл данных: {latest}")
    return latest


def load_data(filepath):
    """Загрузка и базовая фильтрация данных."""
    df = pd.read_csv(filepath)
    logger.info(f"Загружено {len(df)} строк из {filepath}")

    # Проверка обязательных колонок
    required = ['salary_avg', 'skills_list']
    missing = [col for col in required if col not in df.columns]
    if missing:
        available = list(df.columns)
        logger.error(f"Отсутствуют обязательные колонки: {missing}")
        logger.error(f"Доступные колонки: {available}")
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    # Фильтрация: только вакансии с указанной зарплатой
    before = len(df)

    if 'has_salary' in df.columns:
        df = df[df['has_salary'] == True].copy()

    df = df[df['salary_avg'].notna()].copy()
    df = df[df['salary_avg'] > 0].copy()

    after_filter = len(df)
    removed = before - after_filter
    logger.info(
        f"Фильтрация: оставлено {after_filter} строк с зарплатой "
        f"(убрано {removed} без зарплаты / с нулевой)"
    )

    # ---- Удаление выбросов зарплаты ----
    # Абсолютный минимум: ниже 15000₽ — мусорные данные (min=390)
    MIN_SALARY = 15000
    before_abs = len(df)
    df = df[df['salary_avg'] >= MIN_SALARY].copy()
    after_abs = len(df)
    logger.info(
        f"Фильтр abs-min: убрано {before_abs - after_abs} записей < {MIN_SALARY}₽, "
        f"осталось {after_abs}"
    )

    # IQR-метод для верхних выбросов (1.1M и т.д.)
    q1 = df['salary_avg'].quantile(0.25)
    q3 = df['salary_avg'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    before_outlier = len(df)
    df = df[df['salary_avg'] <= upper_bound].copy()
    after_outlier = len(df)

    logger.info(
        f"Удаление выбросов (IQR): верхняя граница {upper_bound:.0f}₽, "
        f"убрано {before_outlier - after_outlier} аномалий, "
        f"осталось {after_outlier}"
    )

    after = after_outlier

    if after == 0:
        raise ValueError(
            "После фильтрации не осталось строк с зарплатой! "
            "Проверьте данные в файле."
        )

    return df


def main():
    """Основная функция обучения моделей."""
    parser = argparse.ArgumentParser(
        description='Обучение ML-моделей предсказания зарплаты'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Режим сравнения всех моделей'
    )
    parser.add_argument(
        '--skip-arima',
        action='store_true',
        help='Пропустить ARIMA (не используется в текущей версии)'
    )
    args = parser.parse_args()

    try:
        # ====== 1. Загрузка данных ======
        logger.info("=" * 60)
        logger.info("ЗАГРУЗКА ДАННЫХ")
        logger.info("=" * 60)

        data_path = find_data_file()
        df = load_data(data_path)

        # ====== 2. Генерация признаков ======
        logger.info("=" * 60)
        logger.info("ГЕНЕРАЦИЯ ПРИЗНАКОВ")
        logger.info("=" * 60)

        fe = FeatureEngineer(top_n_skills=20, tfidf_max_features=20)
        features, top_skill_cols = fe.fit_transform(df)

        logger.info(f"Всего признаков: {len(features.columns)}")
        if top_skill_cols:
            logger.info(
                f"Бинарных фичей навыков: {len(top_skill_cols)} -> "
                f"{top_skill_cols[:5]}..."
            )

        # ====== 3. Целевая переменная ======
        y = df['salary_avg']

        logger.info(
            f"Целевая переменная (salary_avg): "
            f"min={y.min():.0f}, max={y.max():.0f}, "
            f"mean={y.mean():.0f}, median={y.median():.0f}"
        )

        # ====== 4. Обучение моделей ======
        logger.info("=" * 60)
        if args.compare:
            logger.info("РЕЖИМ СРАВНЕНИЯ МОДЕЛЕЙ")
        else:
            logger.info("РЕЖИМ ОБУЧЕНИЯ ЛУЧШЕЙ МОДЕЛИ")
        logger.info("=" * 60)

        predictor = SalaryPredictor()
        predictor.feature_engineer = fe

        if args.compare:
            metrics = predictor.compare(
                features, y, top_skill_cols=top_skill_cols
            )

            # ====== 5. Итоговая таблица ======
            logger.info("=" * 60)
            logger.info("ИТОГИ СРАВНЕНИЯ")
            logger.info("=" * 60)
            header = f"{'Модель':<15} {'R²':>8} {'MAE':>8} {'MAE%':>8} {'RMSE':>8}"
            logger.info(header)
            logger.info("-" * len(header))
            for name, m in metrics.items():
                logger.info(
                    f"{name:<15} {m['R2']:>8.4f} {m['MAE']:>8.0f} "
                    f"{m['MAE%']:>7.1f}% {m['RMSE']:>8.0f}"
                )
        else:
            metrics = predictor.train(
                features, y, top_skill_cols=top_skill_cols
            )

        # ====== 6. Сохранение модели ======
        model_path = PROJECT_ROOT / 'models' / 'salary_predictor.pkl'
        predictor.save(str(model_path))

        logger.info("=" * 60)
        logger.info(f"Лучшая модель: {predictor.best_model_name}")
        logger.info(f"Модель сохранена: {model_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
