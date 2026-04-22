import streamlit as st


def view_aggregation(service):
    st.markdown("<h1 style='text-align: center;'>АГРЕГАЦИЯ ВАКАНСИЙ</h1>", unsafe_allow_html=True)
    st.caption("Обзор рынка труда в реальном времени")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Регион", ["Worldwide", "USA", "Europe"], key="agg_f1")
        c2.selectbox("Период", ["За месяц", "За год", "За все время"], key="agg_f2")
        c3.selectbox("Источник", ["LinkedIn", "Indeed", "Glassdoor"], key="agg_f3")

    st.write("")

    stats = service.get_aggregation_stats()

    k1, k2, k3 = st.columns(3)

    with k1:
        with st.container(border=True):
            st.metric("Средняя зарплата", stats.avg_salary, delta="+5%")
    with k2:
        with st.container(border=True):
            st.metric("Количество вакансий", f"{stats.total_vacancies:,}")
    with k3:
        with st.container(border=True):
            st.metric("Изменения за год", stats.yearly_change)

    st.write("")

    charts = service.get_aggregation_charts()

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            st.subheader("Распределение вакансий")
            st.line_chart(charts.trend_chart, height=300)
    with g2:
        with st.container(border=True):
            st.subheader("Ценные навыки")
            st.bar_chart(charts.skills_chart, height=300)