import streamlit as st
import sys
import os

# --- 1. НАСТРОЙКА ПУТЕЙ (ЧТОБЫ ВИДЕТЬ SRC) ---
# Получаем путь к папке проекта (на уровень выше папки app)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# --- 2. ИМПОРТЫ (Теперь они работают) ---
from src.services.mock import MockService
from app.views.home import view_home
from app.views.aggregation import view_aggregation
from app.views.clusters import view_clusters
from app.views.predict_salary import view_salary_predictor

# --- 3. КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="DataTrack",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS: СКРЫВАЕМ ВСЕ ЛИШНЕЕ ---
st.markdown("""
<style>
    /* 1. Скрываем боковое меню (Sidebar) */
    [data-testid="stSidebarNav"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

    /* 2. Скрываем верхнюю полосу (Header) с кнопкой Deploy и меню */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* 3. Настраиваем отступы, чтобы контент начинался выше */
    /* block-container — это основной контейнер страницы */
    .block-container {
        padding-top: 2rem; /* Оставляем немного места, чтобы не прилипало к краю браузера */
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 4. КОМПОНЕНТ НАВИГАЦИИ (TOP BAR) ---
def render_top_nav():
    with st.container():
        # Сетка: Логотип (1 часть) | Кнопки меню (5 частей)
        col_logo, col_nav = st.columns([1, 5])

        with col_logo:
            st.markdown("### 🦁 DataTrack")

        with col_nav:
            # 4 Кнопки
            nav_home, nav_agg, nav_clus, nav_sal = st.columns(4)

            # Текущая активная страница
            current_page = st.session_state.get('page', 'home')

            # --- КНОПКИ ---
            # Если страница активна, кнопка 'primary' (красная/выделенная), иначе 'secondary'

            if nav_home.button("🏠 Главная", use_container_width=True,
                               type="primary" if current_page == 'home' else "secondary"):
                st.session_state['page'] = 'home'
                st.rerun()

            if nav_agg.button("📊 Агрегация", use_container_width=True,
                              type="primary" if current_page == 'aggregation' else "secondary"):
                st.session_state['page'] = 'aggregation'
                st.rerun()

            if nav_clus.button("🧩 Кластеры", use_container_width=True,
                               type="primary" if current_page == 'clusters' else "secondary"):
                st.session_state['page'] = 'clusters'
                st.rerun()

            if nav_sal.button("💰 Прогноз ЗП", use_container_width=True,
                              type="primary" if current_page == 'salary' else "secondary"):
                st.session_state['page'] = 'salary'
                st.rerun()

        st.divider()


# --- 5. ТОЧКА ВХОДА (MAIN) ---
def main():
    # Инициализация состояния (Routing)
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    # --- ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ ---
    # Создаем экземпляры классов, которые ты прислал
    # StatisticsService ищет реальные файлы
    # MockService генерирует фейковые данные
    mock_service = MockService()

    # Рисуем меню (оно всегда сверху)
    render_top_nav()

    # --- РОУТИНГ (ПЕРЕКЛЮЧЕНИЕ) ---
    page = st.session_state['page']

    if page == 'home':
        # Главная страница получает РЕАЛЬНЫЙ сервис
        view_home(mock_service)

    elif page == 'aggregation':
        # Остальные получают MOCK сервис
        view_aggregation(mock_service)

    elif page == 'clusters':
        view_clusters(mock_service)

    elif page == 'salary':
        view_salary_predictor(mock_service)


if __name__ == "__main__":
    main()