import streamlit as st

from src.infrastructure.vacancies_repository import VacanciesRepository
from src.services.home_service import HomeService


@st.cache_resource
def _get_service(data_checksum: str) -> HomeService:
    # Keyed by checksum: when a new CSV appears, the checksum changes and a
    # fresh service is built on next run. Same pattern as clusters.py.
    return HomeService(VacanciesRepository())


@st.cache_data(ttl=300)
def _get_checksum() -> str:
    return VacanciesRepository().compute_checksum()


def view_home(service=None):
    st.markdown("<h1 style='text-align: center;'>DataTrack Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("Добро пожаловать в DataTrack!")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **О проекте**

        **DataTrack** — аналитический веб-сервис для исследования рынка труда специалистов
        по анализу данных на основе данных hh.ru. Платформа позволяет изучить рынок
        с разных сторон: от сводной статистики и интерактивных графиков до кластеризации
        вакансий и машинного обучения.

        **Возможности системы:**
        1. 📊 **Агрегация** — фильтрация вакансий по региону, опыту, зарплате, роли и навыкам
           с визуализацией в шести вкладках. Вкладка «Мой профиль» показывает ваш перцентиль
           на рынке и рекомендует навыки для роста зарплаты. Доступен экспорт в CSV.
        2. 🧩 **Кластеризация** — группировка вакансий по выбранным признакам.
           Алгоритм подбирается автоматически: K-Means для числовых данных,
           K-Modes для категориальных, K-Prototypes для смешанных.
        3. 📈 **Тренды (SARIMA)** — анализ динамики числа вакансий и зарплат во времени
           с SARIMA-прогнозом и доверительными интервалами.
        4. 💰 **Прогноз зарплат** — оценка рыночной зарплаты по вашему профилю
           с помощью CatBoost. Показывает важность признаков и сравнивает несколько моделей.
        """)

    with col2:
        st.info("""
        **Команда проекта:**
        НИУ ВШЭ, ФКН · 2026
        """)

    st.markdown("---")
    st.subheader("Сводка по базе данных")

    checksum = _get_checksum()
    home_service = _get_service(checksum)
    stats = home_service.get_home_statistics()

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
        if val >= 100000:
            display = f"{val // 1000}K ₽"
        elif val > 0:
            display = f"{val:,} ₽".replace(",", " ")
        else:
            display = "—"
        st.metric("Средняя ЗП", display)

    with m4:
        val = stats.active_vacancies
        st.metric("Активных (30 дн)", f"{val:,}".replace(",", " "))

    st.caption(f"📅 Обновлено: {stats.last_updated}")

    st.markdown("---")
    if st.button("🔄 Обновить статистику", use_container_width=True):
        _get_checksum.clear()
        _get_service.clear()
        st.rerun()
