import streamlit as st

def view_clusters(service):
    st.markdown("<h1 style='text-align: center;'>КЛАСТЕРЫ ВАКАНСИЙ</h1>", unsafe_allow_html=True)

    # Search & Filter bar
    with st.container(border=True):
        c1, c2, c3, btn = st.columns([3, 1, 1, 0.5])
        c1.text_input("Поиск", placeholder="Поиск кластера...", label_visibility="collapsed")
        c2.selectbox("Фильтр 1", ["Все"], label_visibility="collapsed")
        c3.selectbox("Фильтр 2", ["Все"], label_visibility="collapsed")
        btn.button("🔍")

    clusters = service.get_clusters()

    # Grid 3x2
    cols_per_row = 3
    rows = [clusters[i:i + cols_per_row] for i in range(0, len(clusters), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, cluster in enumerate(row):
            with cols[idx]:
                with st.container(border=True):
                    st.subheader(cluster.title)
                    st.write(cluster.description)
                    st.divider()
                    m1, m2 = st.columns(2)
                    m1.metric("Вакансий", cluster.vacancies_count)
                    m2.metric("ЗП", cluster.avg_salary)
                    st.divider()
                    st.caption("Stack: " + ", ".join(cluster.skills))