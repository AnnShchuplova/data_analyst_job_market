import streamlit as st


def view_home(service):
    st.markdown("<h1 style='text-align: center;'>DataTrack Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("Добро пожаловать в DataTrack!")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **О проекте**
        **DataTrack** - программный комплекс для анализа рынка труда.

        **Возможности системы:**
        1. 📊 **Агрегация вакансий** - сбор и визуализация
        2. 🧩 **Кластеризация** - группировка вакансий
        3. 💰 **Прогноз зарплат** - ML-оценка стоимости специалиста
        """)

    with col2:
        st.info("""
        **Команда проекта:**
        НИУ ВШЭ, ФКН • 2025
        """)

    st.markdown("---")
    st.subheader("Сводка по базе данных")

    stats = service.get_home_statistics()

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        val = stats.total_vacancies
        st.metric("Собрано вакансий", f"{val:,}".replace(",", " "))

    with m2:
        val = stats.with_salary
        total = stats.total_vacancies
        perc = (val / total * 100) if total > 0 else 0
        st.metric("С указанной ЗП", f"{val:,}".replace(",", " "), f"{perc:.0f}%")

    with m3:
        val = stats.avg_salary
        display = f"{val // 1000}K ₽" if val > 100000 else f"{val} ₽"
        st.metric("Средняя ЗП", display)

    with m4:
        val = stats.active_vacancies
        st.metric("Активных (30 дн)", f"{val:,}".replace(",", " "))

    st.caption(f"📅 Обновлено: {stats.last_updated}")

    st.markdown("---")
    if st.button("🔄 Обновить статистику", use_container_width=True):
        st.rerun()