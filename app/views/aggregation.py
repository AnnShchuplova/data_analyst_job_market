import streamlit as st


def view_aggregation(service):
    st.markdown("<h1 style='text-align: center;'>АГРЕГАЦИЯ ВАКАНСИЙ</h1>", unsafe_allow_html=True)
    st.caption("Обзор рынка труда в реальном времени")

    # 1. Фильтры (Top Row)
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Регион", ["Worldwide", "USA", "Europe"], key="agg_f1")
        c2.selectbox("Период", ["За месяц", "За год", "За все время"], key="agg_f2")
        c3.selectbox("Источник", ["LinkedIn", "Indeed", "Glassdoor"], key="agg_f3")

    st.write("")

    # 2. KPI Карточки (Middle Row)
    # Получаем объект AggregationStats
    stats = service.get_aggregation_stats()

    k1, k2, k3 = st.columns(3)

    with k1:
        with st.container(border=True):
            # Обращение через ТОЧКУ
            st.metric("Средняя зарплата", stats.avg_salary, delta="+5%")
    with k2:
        with st.container(border=True):
            # Обращение через ТОЧКУ
            st.metric("Количество вакансий", f"{stats.total_vacancies:,}")
    with k3:
        with st.container(border=True):
            # Обращение через ТОЧКУ
            st.metric("Изменения за год", stats.yearly_change)

    st.write("")

    # 3. Графики (Bottom Row)
    # --- ИСПРАВЛЕНИЕ ОШИБКИ ТУТ ---

    # БЫЛО: chart1, chart2 = service.get_aggregation_charts()

    # СТАЛО: Получаем объект AggregationCharts
    charts = service.get_aggregation_charts()

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.subheader("Распределение вакансий")
            # Берем данные из поля объекта
            st.line_chart(charts.trend_chart, height=300)
    with g2:
        with st.container(border=True):
            st.subheader("Ценные навыки")
            # Берем данные из поля объекта
            st.bar_chart(charts.skills_chart, height=300)