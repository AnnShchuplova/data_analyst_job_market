import streamlit as st
import time

st.set_page_config(page_title="Прогноз зарплат", layout="wide")

st.title("💰 Предсказатель заработной платы")
st.markdown("---")

st.info("""
> «Предсказание заработной платы»

**Модель машинного обучения для прогнозирования ЗП аналитика данных**
""")

with st.form("salary_prediction_form"):
    st.subheader("📝 Параметры вакансии")
    
    col1, col2 = st.columns(2)
    
    with col1:
        experience = st.selectbox(
            "Опыт работы",
            ["Junior (до 1 года)", "Middle (1-3 года)", "Senior (3+ года)"],
            index=1
        )
        
        region = st.selectbox(
            "Регион",
            ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Другой"],
            index=0
        )
    
    with col2:
        skills = st.multiselect(
            "Ключевые навыки",
            [
                "Python", "SQL", "Excel", "Tableau", "Power BI",
                "Statistics", "Machine Learning", "A/B testing",
                "Data Visualization", "ETL"
            ],
            default=["Python", "SQL", "Excel"]
        )
        
        position = st.text_input("Должность", value="Аналитик данных")
    
    submitted = st.form_submit_button("🎯 Предсказать зарплату")
    
    if submitted:
        with st.spinner("Выполняю прогноз..."):
            time.sleep(1)  # здесь будет вызов ML-модели
            
            # Результат
            st.success("### Прогноз заработной платы")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Прогноз", "95 000 ₽") # пока рандомные значение
            
            with col2:
                st.metric("Диапазон", "80 000 - 110 000 ₽")
            
            with col3:
                st.metric("Доверительный интервал", "85%")
            
            st.markdown("---")
            st.subheader("📊 Факторы влияния на зарплату")
            
            factors = {
                "Регион (Москва)": "+25%",
                "Опыт (Middle)": "+15%", 
                "Навык Python": "+10%",
                "Навык SQL": "+8%"
            }
            
            for factor, impact in factors.items():
                st.write(f"- {factor}: {impact}")

if __name__ == "__main__":
    st.caption("""
    (Модели еще нет в системе)
    """)