import streamlit as st
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.services.mock import MockService
from app.views.home import view_home
from app.views.aggregation import view_aggregation
from app.views.clusters import view_clusters
from app.views.predict_salary import view_salary_predictor
from app.views.arima import view_arima

st.set_page_config(
    page_title="DataTrack",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Прячем sidebar/nav/header ДО первого рендера, чтобы не мигали.
       !important + visibility + width:0 подавляют все варианты,
       которые Streamlit успевает показать до подгрузки стилей. */
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    /* Контент занимает всю ширину (без резерва под sidebar) */
    [data-testid="stAppViewContainer"] > .main {
        margin-left: 0 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def render_top_nav():
    with st.container():
        col_logo, col_nav = st.columns([1, 5])

        with col_logo:
            st.markdown("### 🗂️ DataTrack")

        with col_nav:
            nav_home, nav_agg, nav_clus, nav_arima, nav_sal = st.columns(5)

            current_page = st.session_state.get('page', 'home')

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

            if nav_arima.button("📈 Тренды (SARIMA)", use_container_width=True,
                                type="primary" if current_page == 'arima' else "secondary"):
                st.session_state['page'] = 'arima'
                st.rerun()

            if nav_sal.button("💰 Прогноз ЗП", use_container_width=True,
                              type="primary" if current_page == 'salary' else "secondary"):
                st.session_state['page'] = 'salary'
                st.rerun()

        st.divider()
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    mock_service = MockService()

    render_top_nav()

    page = st.session_state['page']

    if page == 'home':
        view_home(mock_service)

    elif page == 'aggregation':
        view_aggregation(mock_service)

    elif page == 'clusters':
        view_clusters()

    elif page == 'arima':
        view_arima()

    elif page == 'salary':
        view_salary_predictor(mock_service)


if __name__ == "__main__":
    main()