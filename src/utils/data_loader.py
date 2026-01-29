import os
import glob
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def simplify_job_name(name):
    if not isinstance(name, str):
        return 'Other'

    name_lower = name.lower()

    categories = {
        'BI Analyst': ['bi', 'tableau', 'power bi', 'qlik', 'looker'],
        'System Analyst': ['system', 'системн'],
        'Business Analyst': ['business', 'бизнес'],
        'Financial Analyst': ['financial', 'финанс'],
        'Data Analyst': ['data', 'данны', 'product', 'продуктов'],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name_lower:
                if keyword == 'bi' and ('big data' in name_lower or 'mobile' in name_lower):
                    continue
                return category

    return 'Other'

def load_vacancies_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'processed')

    search_pattern = os.path.join(data_dir, "new_cleaned_vacancies*.csv")
    files = glob.glob(search_pattern)

    if not files:
        logger.error(f"Files not found: {search_pattern}")
        return pd.DataFrame()

    latest_file = max(files, key=os.path.getmtime)

    try:
        df = pd.read_csv(latest_file)

        useful_cols = [
            'id',
            'name',
            'salary_avg',
            'avg_experience_years',
            'min_experience_years',
            'requirement',
            'responsibility',
            'schedule_name',
            'employment_name',
            'employer_name',
            'area_name',
            'skills_list'
        ]

        df_ml = df[useful_cols].copy()

        df_ml['name'] = df_ml['name'].apply(simplify_job_name)

        return df_ml

    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_vacancies_data()