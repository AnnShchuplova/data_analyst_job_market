import pandas as pd
import os
import numpy as np
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        pass
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка JSON данных в DataFrame"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        logger.info(f"Загружено {len(df)} вакансий аналитиков")
        return df
    
    def clean_salary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очистка данных о зарплате
        Сохраняет ВСЕ вакансии, даже без зарплаты
        """
        df = df.copy()
        
        if 'salary' in df.columns:
            df['salary_from'] = df['salary'].apply(
                lambda x: x.get('from') if isinstance(x, dict) else None
            )
            df['salary_to'] = df['salary'].apply(
                lambda x: x.get('to') if isinstance(x, dict) else None
            )
            df['salary_currency'] = df['salary'].apply(
                lambda x: x.get('currency') if isinstance(x, dict) else None
            )
        
        # df = df[df['salary_currency'] == 'RUR'] 
        
        df['salary_from'] = pd.to_numeric(df['salary_from'], errors='coerce')
        df['salary_to'] = pd.to_numeric(df['salary_to'], errors='coerce')
    
        
        df['has_salary'] = (~df['salary_from'].isna()) | (~df['salary_to'].isna())
        
        df['salary_avg'] = df.apply(
            lambda row: self._calculate_avg_salary(row['salary_from'], row['salary_to']),
            axis=1
        )
       
        total = len(df)
        with_salary = df['has_salary'].sum()
        logger.info(f"Зарплата: {with_salary} вакансий с ЗП из {total} ({with_salary/total*100:.1f}%)")
        
        return df
    
    def _calculate_avg_salary(self, salary_from, salary_to):
       
        if pd.isna(salary_from) and pd.isna(salary_to):
            return np.nan  
        elif pd.isna(salary_from):
            return salary_to
        elif pd.isna(salary_to):
            return salary_from
        else:
            return (salary_from + salary_to) / 2
    
    def clean_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'experience' in df.columns:
            df['experience_id'] = df['experience'].apply(
                lambda x: x.get('id') if isinstance(x, dict) else None
            )
            df['experience_name'] = df['experience'].apply(
                lambda x: x.get('name') if isinstance(x, dict) else None
            )
            
            
            def estimate_years(exp_id):
                if exp_id == 'noExperience':
                    return 0
                elif exp_id == 'between1And3':
                    return 2  
                elif exp_id == 'between3And6':
                    return 4.5  
                elif exp_id == 'moreThan6':
                    return 8 
                else:
                    return np.nan
            
            df['estimated_experience_years'] = df['experience_id'].apply(estimate_years)
        
        if 'experience_name' in df.columns:
            exp_counts = df['experience_name'].value_counts()
            logger.info(f"Опыт работы: {exp_counts.to_dict()}")
        
        return df
    
    def extract_skills(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        if 'key_skills' in df.columns:
            df['skills_list'] = df['key_skills'].apply(
                lambda x: [skill.get('name') for skill in x] 
                if isinstance(x, list) 
                else []
            )
            
            df['skills_count'] = df['skills_list'].apply(len)
           
            if df['skills_list'].notna().sum() > 10:
                top_skills = self._get_top_skills(df, top_n=15)
                for skill in top_skills:
                    col_name = f'skill_{skill.replace(" ", "_").replace("/", "_").lower()}'
                    df[col_name] = df['skills_list'].apply(
                        lambda x: 1 if skill in x else 0
                    )
        return df
    
    def _get_top_skills(self, df, top_n=15):
        all_skills = []
        for skills in df['skills_list'].dropna():
            all_skills.extend(skills)
        
        from collections import Counter
        return [skill for skill, count in Counter(all_skills).most_common(top_n)]
    
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
    
        if 'published_at' in df.columns:
            df['published_at'] = pd.to_datetime(df['published_at'])
        
            df['published_date'] = df['published_at'].dt.date
            df['published_year_month'] = df['published_at'].dt.strftime('%Y-%m')
        
            df['days_since_publication'] = (pd.Timestamp.now(tz='UTC') - df['published_at']).dt.days
    
        return df
    
    def clean_employer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'employer' in df.columns:
            df['employer_name'] = df['employer'].apply(
                lambda x: x.get('name') if isinstance(x, dict) else None
            )
            df['employer_id'] = df['employer'].apply(
                lambda x: x.get('id') if isinstance(x, dict) else None
            )
        
        return df
    
    def clean_area(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'area' in df.columns:
            df['area_name'] = df['area'].apply(
                lambda x: x.get('name') if isinstance(x, dict) else None
            )
            df['area_id'] = df['area'].apply(
                lambda x: x.get('id') if isinstance(x, dict) else None
            )
        
        return df
    
    def remove_outliers_optional(self, df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:

        if not remove_outliers or 'salary_avg' not in df.columns:
            return df
        
        df = df.copy()
        
        salary_data = df['salary_avg'].dropna()
        
        if len(salary_data) > 10: 
            Q1 = salary_data.quantile(0.25)
            Q3 = salary_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = ((df['salary_avg'] >= lower_bound) & (df['salary_avg'] <= upper_bound)) | df['salary_avg'].isna()
            
            before = len(df)
            df = df[mask]
            after = len(df)
            
            logger.info(f"Выбросы: удалено {before - after} вакансий с аномальной ЗП")
    
        return df
    
    def run_full_clean(self, input_file: str, output_file: str = None, remove_outliers: bool = False):
        
        logger.info(f"Начало очистки: {input_file}")
        
        df = self.load_data(input_file)
        
        df = self.clean_salary(df)    
        df = self.clean_experience(df) 
        df = self.extract_skills(df)    
        df = self.clean_dates(df)       
        df = self.clean_employer(df)    
        df = self.clean_area(df)       
    
        df = self.remove_outliers_optional(df, remove_outliers)

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/processed/cleaned_vacancies_{timestamp}.csv"
        
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Очищенные данные сохранены: {output_file}")
        
        self._print_statistics(df, input_file, output_file)
        
        return df
    
    def _print_statistics(self, df, input_file, output_file):
    
        if 'has_salary' in df.columns:
            with_salary = df['has_salary'].sum()
            without_salary = len(df) - with_salary
            print(f"С указанной зарплатой: {with_salary} ({with_salary/len(df)*100:.1f}%)")
            print(f"Без указанной зарплаты: {without_salary} ({without_salary/len(df)*100:.1f}%)")
        
        if 'salary_avg' in df.columns:
            salary_data = df['salary_avg'].dropna()
            if len(salary_data) > 0:
                print(f"Средняя зарплата: {salary_data.mean():.0f} руб.")
                print(f"Медианная зарплата: {salary_data.median():.0f} руб.")
                print(f"Мин/Макс: {salary_data.min():.0f} / {salary_data.max():.0f} руб.")
        
        if 'experience_name' in df.columns:
            print(f"\nРаспределение по опыту:")
            for level, count in df['experience_name'].value_counts().items():
                print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'skills_count' in df.columns:
            avg_skills = df['skills_count'].mean()
            print(f"\nСреднее количество навыков: {avg_skills:.1f}")
        
        if 'skills_list' in df.columns:
            all_skills = []
            for skills in df['skills_list'].dropna():
                all_skills.extend(skills)
            
            if all_skills:
                from collections import Counter
                top_skills = Counter(all_skills).most_common(10)
                print(f"\nТоп-10 навыков:")
                for skill, count in top_skills:
                    percentage = count / len(df) * 100
                    print(f"  {skill}: {count} ({percentage:.1f}%)")
        


if __name__ == "__main__":
    cleaner = DataCleaner()
    
    input_file = "data/processed/analyst_vacancies_20260123_153601.json"
    
    if not os.path.exists(input_file):
        print(f"Файл не найден: {input_file}")
    else:
        df_cleaned = cleaner.run_full_clean(input_file, remove_outliers=False)
        print("\nОчистка завершена")