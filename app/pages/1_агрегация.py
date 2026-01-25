import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Агрегация вакансий", layout="wide")

st.title("📊 Агрегированные данные по вакансиям")
st.markdown("---")

def load_data():
    try:
        processed_dir = "data/processed"
        if os.path.exists(processed_dir):
            files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
                file_path = os.path.join(processed_dir, latest_file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
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
        st.success(f"✅ Данные загружены: {filename}")
        
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
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if 'salary_avg' in df.columns:
                    fig2 = px.histogram(
                        df, 
                        x='salary_avg',
                        nbins=20,
                        title="Распределение зарплат",
                        labels={'salary_avg': 'Средняя зарплата, руб.'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        
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
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            st.subheader("Географическое распределение")
            
            if 'area' in df.columns:
                region_counts = df['area'].apply(lambda x: x.get('name') if isinstance(x, dict) else 'Не указан').value_counts()
                
                fig4 = px.bar(
                    x=region_counts.index[:10],
                    y=region_counts.values[:10],
                    title="Топ-10 регионов по количеству вакансий",
                    labels={'x': 'Регион', 'y': 'Количество вакансий'}
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab4:
            st.subheader("Таблица данных")
            st.dataframe(df[['name', 'salary_avg', 'experience_name']].head(20), use_container_width=True)
    
    else:
        st.warning("""
        ## 📥 Данные не найдены
        """)
        
        if st.button("📊 Загрузить тестовые данные"):
            st.info("Тестовые данные загружены (демо-режим)")

if __name__ == "__main__":
    main()