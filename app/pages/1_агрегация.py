import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Агрегация вакансий", layout="wide")

st.title("📊 Агрегированные данные по вакансиям")
st.markdown("---")

def load_data():
    """Загрузка данных из finaldata/ (формат CSV)"""
    try:
        # Используем абсолютный путь относительно корня проекта
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        processed_dir = os.path.join(project_root, "finaldata")
        if os.path.exists(processed_dir):
            # Ищем CSV файлы (cleaner.py сохраняет в CSV)
            files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
                file_path = os.path.join(processed_dir, latest_file)
                
                df = pd.read_csv(file_path, encoding='utf-8')
                return df, latest_file
        
        return None, None
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None, None

def main():
   
    with st.sidebar:
        st.header("⚙️ Параметры анализа")
        
        regions = ["Все регионы", "Москва", "Санкт-Петербург", "Другие"]
        selected_region = st.selectbox("Регион", regions)
        
        salary_range = st.slider("Диапазон зарплат (тыс. руб.)", 0, 500, (0, 300))
        
        experience = st.multiselect(
            "Опыт работы",
            ["Нет опыта", "1-3 года", "3-6 лет", "Более 6 лет"],
            default=["1-3 года", "3-6 лет"]
        )
        
        update_btn = st.button("🔄 Обновить данные", type="primary")
    
    df, filename = load_data()
    
    if df is not None:
        # Применяем фильтры
        df_filtered = df.copy()
        
        if selected_region != "Все регионы":
            if selected_region == "Другие":
                if 'area_name' in df_filtered.columns:
                    major_regions = ["Москва", "Санкт-Петербург"]
                    df_filtered = df_filtered[~df_filtered['area_name'].isin(major_regions)]
            else:
                if 'area_name' in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered['area_name'] == selected_region]
        
        if salary_range != (0, 500):
            if 'salary_avg' in df_filtered.columns:
                df_filtered = df_filtered[
                    (df_filtered['salary_avg'].fillna(0) / 1000 >= salary_range[0]) &
                    (df_filtered['salary_avg'].fillna(999) / 1000 <= salary_range[1])
                ]
        
        if experience:
            exp_map = {
                "Нет опыта": "noExperience",
                "1-3 года": "between1And3",
                "3-6 лет": "between3And6",
                "Более 6 лет": "moreThan6"
            }
            exp_ids = [exp_map[e] for e in experience if e in exp_map]
            if exp_ids and 'experience_id' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['experience_id'].isin(exp_ids)]
        
        df = df_filtered
        st.success(f"✅ Данные загружены: {filename} ({len(df)} записей после фильтрации)")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Обзор", "💰 Зарплаты", "🏙️ Регионы", "📋 Таблица"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
               
                if 'experience_name' in df.columns:
                    exp_counts = df['experience_name'].value_counts()
                    fig1 = px.pie(
                        values=exp_counts.values,
                        names=exp_counts.index,
                        title="Распределение по опыту работы"
                    )
                    st.plotly_chart(fig1, width="stretch")
            
            with col2:
                if 'salary_avg' in df.columns:
                    fig2 = px.histogram(
                        df, 
                        x='salary_avg',
                        nbins=20,
                        title="Распределение зарплат",
                        labels={'salary_avg': 'Средняя зарплата, руб.'}
                    )
                    st.plotly_chart(fig2, width="stretch")
        
        with tab2:
            st.subheader("Анализ зарплат")
            
            if 'salary_avg' in df.columns and 'experience_name' in df.columns:
                fig3 = px.box(
                    df,
                    x='experience_name',
                    y='salary_avg',
                    title="Зарплаты по уровням опыта",
                    labels={'salary_avg': 'Зарплата, руб.', 'experience_name': 'Опыт'}
                )
                st.plotly_chart(fig3, width="stretch")
        
        with tab3:
            st.subheader("Географическое распределение")
            
            # В CSV данные area уже извлечены в area_name (не dict)
            area_col = 'area_name' if 'area_name' in df.columns else 'area'
            if area_col in df.columns:
                if area_col == 'area_name':
                    region_counts = df['area_name'].fillna('Не указан').value_counts()
                else:
                    region_counts = df['area'].apply(lambda x: x.get('name') if isinstance(x, dict) else 'Не указан').value_counts()
                
                fig4 = px.bar(
                    x=region_counts.index[:10],
                    y=region_counts.values[:10],
                    title="Топ-10 регионов по количеству вакансий",
                    labels={'x': 'Регион', 'y': 'Количество вакансий'}
                )
                st.plotly_chart(fig4, width="stretch")
        
        with tab4:
            st.subheader("Таблица данных")
            st.dataframe(df[['name', 'salary_avg', 'experience_name']].head(20), width="stretch")
    
    else:
        st.warning("""
        ## 📥 Данные не найдены
        """)
        
        if st.button("📊 Загрузить тестовые данные"):
            st.info("Тестовые данные загружены (демо-режим)")

if __name__ == "__main__":
    main()