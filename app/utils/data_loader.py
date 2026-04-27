"""
Утилита загрузки данных для Streamlit-приложения.

Предоставляет общий интерфейс для загрузки данных из finaldata/.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_processed_data(project_root: str = None) -> tuple:
    """Загрузка последнего CSV файла из finaldata/.
    
    Returns:
    --------
    (df, filename) : DataFrame и имя файла, или (None, None) если файлы не найдены
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    processed_dir = os.path.join(project_root, "finaldata")
    
    if not os.path.exists(processed_dir):
        logger.warning(f"Папка '{processed_dir}' не существует")
        return None, None
    
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    if not csv_files:
        logger.warning(f"В '{processed_dir}' нет CSV файлов")
        return None, None
    
    latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
    file_path = os.path.join(processed_dir, latest_file)
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Загружено {len(df)} записей из {latest_file}")
        return df, latest_file
    except Exception as e:
        logger.error(f"Ошибка загрузки {latest_file}: {e}")
        return None, None
