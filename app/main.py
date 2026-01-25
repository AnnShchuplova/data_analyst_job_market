import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="DataTrack - Анализ рынка труда",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_latest_data():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(project_root, "data", "processed")
        
        if not os.path.exists(processed_dir):
            st.error(f"Папка '{processed_dir}' не существует")
            return None
        csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not csv_files:
            return None
        
        # последний файл
        latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
        file_path = os.path.join(processed_dir, latest_file)
        
        df = pd.read_csv(file_path, encoding='utf-8')
        return df, latest_file
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None

def calculate_statistics(df):
    """Вычисление статистики из данных"""
    if df is None or len(df) == 0:
        return {
            'total_vacancies': 0,
            'with_salary': 0,
            'avg_salary': 0,
            'active_vacancies': 0,
            'date': '—'
        }
    
    stats = {
        'total_vacancies': len(df)
    }
    
    if 'salary_avg' in df.columns:
        salary_data = df['salary_avg'].dropna()
        stats['with_salary'] = len(salary_data)
        if len(salary_data) > 0:
            stats['avg_salary'] = int(salary_data.mean())
        else:
            stats['avg_salary'] = 0
    else:
        stats['with_salary'] = 0
        stats['avg_salary'] = 0

    if 'published_at' in df.columns:
        try:
            df['published_at'] = pd.to_datetime(df['published_at'])
            month_ago = datetime.now() - timedelta(days=30)
            stats['active_vacancies'] = len(df[df['published_at'] >= month_ago])
        except:
            stats['active_vacancies'] = len(df)
    else:
        stats['active_vacancies'] = len(df)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, "data", "processed")
        
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    if csv_files:
        latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
        file_path = os.path.join(processed_dir, latest_file)
        timestamp = os.path.getmtime(file_path)
        stats['date'] = datetime.fromtimestamp(timestamp).strftime('%d.%m.%Y')
    else:
        stats['date'] = '—'
    
    return stats

def main():
    st.title("DataTrack - Анализ рынка труда аналитиков данных")
    st.markdown("---")
  
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/data-configuration.png", width=80)
        st.title("DataTrack v1.0")
        st.markdown("Программа для анализа рынка труда аналитиков данных")
        st.markdown("---")
        
        st.markdown("Навигация")
        st.markdown("Используйте меню слева для перехода между страницами")
        
        st.markdown("---")
        st.caption("НИУ ВШЭ • Программная инженерия • 2025")
    
    st.header("Добро пожаловать в DataTrack!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        О проекте
        
        **DataTrack** - это программный комплекс для анализа рынка труда 
        в области аналитики данных на основе данных с платформы HeadHunter.
        
        Возможности системы:
        
        1. **Агрегация вакансий** - сбор и визуализация данных
        2. **Кластеризация** - автоматическая группировка вакансий
        3. **Прогноз зарплат** - ML-модель для предсказания ЗП
        4. **Анализ трендов** - временные ряды (ARIMA)
        
        Быстрый старт
        
        1. Перейдите на страницу **Агрегация**
        2. Загрузите или соберите данные
        3. Проанализируйте рынок труда
        """)
    
    with col2:
        st.info("""
        Команда проекта:
        - Щуплова А.И. (БПИ237)
        - Тищенко Н.А. (БПИ247)
        
        Курсовой проект
        НИУ ВШЭ, ФКН
        """)
    
    st.markdown("---")
    st.subheader("Статистика")
    
    data_result = load_latest_data()
    if data_result:
        df, filename = data_result
        stats = calculate_statistics(df)
    else:
        stats = calculate_statistics(None)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = stats['total_vacancies']
        if total > 0:
            display_total = f"{total:,}".replace(",", " ")
            st.metric("Собрано вакансий", display_total)
        else:
            st.metric("Собрано вакансий", "0")
    
    with col2:
        with_salary = stats['with_salary']
        if with_salary > 0:
            display_salary = f"{with_salary:,}".replace(",", " ")
            percentage = (with_salary / total * 100) if total > 0 else 0
            st.metric("С указанной ЗП", display_salary, f"{percentage:.0f}%")
        else:
            st.metric("С указанной ЗП", "0")
    
    with col3:
        avg_salary = stats['avg_salary']
        if avg_salary > 0:
            if avg_salary >= 100000:
                display_salary = f"{avg_salary//1000}K ₽"
            else:
                display_salary = f"{avg_salary:,} ₽".replace(",", " ")
            st.metric("Средняя ЗП", display_salary)
        else:
            st.metric("Средняя ЗП", "—")
    
    with col4:
        active = stats['active_vacancies']
        if active > 0:
            display_active = f"{active:,}".replace(",", " ")
            st.metric("Вакансий", display_active)
        else:
            st.metric("Вакансий", "0")

    st.caption(f"Данные обновлены: {stats['date']}")
    

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Обновить статистику", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("Собрать новые данные", use_container_width=True):
            st.info("Для сбора данных запустите: python scripts/collect_data.py")

if __name__ == "__main__":
    main()