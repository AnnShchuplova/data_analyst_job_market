import streamlit as st
import plotly.express as px
from src.utils.data_loader import load_vacancies_data
from src.services.clustering_service import ClusteringService


@st.cache_resource
def get_service():
    df = load_vacancies_data()
    return ClusteringService(df) if not df.empty else None


def view_clusters(mock_service=None):
    service = get_service()

    st.markdown("<h1 style='text-align: center;'>🧩 УМНАЯ КЛАСТЕРИЗАЦИЯ</h1>", unsafe_allow_html=True)

    if service is None or service.df.empty:
        st.error("❌ Не удалось загрузить данные. Проверьте папку data/processed.")
        return
    with st.container(border=True):
        st.subheader("⚙️ Параметры")
        c1, c2 = st.columns([1, 1])
        with c1:
            selected_features = st.multiselect(
                "Признаки для группировки:",
                options=["Зарплата", "Минимальный опыт", "Название вакансии", "График работы", "Местность"],
                default=["Зарплата", "Название вакансии"]
            )
        with c2:
            k_range = st.slider("Диапазон кластеров:", 2, 20, (2, 12))

        run_btn = st.button("🚀 Запустить анализ", use_container_width=True, type="primary",
                            disabled=len(selected_features) == 0)
    if run_btn:
        with st.spinner("🧠 Анализируем рынок..."):
            try:
                res = service.perform_clustering(selected_features, range(k_range[0], k_range[1] + 1))
                st.session_state['cluster_result'] = res
            except Exception as e:
                st.error(f"Ошибка: {e}")

    if 'cluster_result' in st.session_state:
        res = st.session_state['cluster_result']

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Алгоритм", res.method_name)
        m2.metric("Кластеров", res.n_clusters)
        m3.metric("Silhouette Score", f"{res.silhouette_score:.3f}")

        if res.k_scores:
            with st.expander("📈 Выбор оптимального K (Silhouette)", expanded=False):
                ks = [s[0] for s in res.k_scores]
                scores = [s[1] for s in res.k_scores]
                fig = px.line(
                    x=ks, y=scores,
                    markers=True,
                    labels={"x": "K (число кластеров)", "y": "Silhouette Score"},
                    title="Silhouette Score по числу кластеров"
                )
                fig.add_vline(
                    x=res.n_clusters,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Выбрано K={res.n_clusters}",
                    annotation_position="top right"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        clusters = res.clusters
        rows = [clusters[i:i + 3] for i in range(0, len(clusters), 3)]

        for row in rows:
            cols = st.columns(3)
            for idx, cluster in enumerate(row):
                with cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"#### {cluster.title}")
                        st.caption(cluster.description)
                        st.divider()

                        k1, k2 = st.columns(2)
                        k1.metric("Вакансий", cluster.vacancies_count)
                        k2.metric("Ср. ЗП", cluster.avg_salary)

                        rem_pct = cluster.remote_rate
                        off_pct = 100 - rem_pct

                        st.markdown(f"""
                        <div style="margin-top: 10px; margin-bottom: 5px; font-size: 0.8em; display: flex; justify-content: space-between;">
                            <span style="color: #666;">🏢 Офис {int(off_pct)}%</span>
                            <span style="color: #00CC96;">🌍 Удаленка {int(rem_pct)}%</span>
                        </div>
                        <div style="width: 100%; height: 8px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden; display: flex;">
                            <div style="width: {off_pct}%; background-color: #d1d5db;"></div>
                            <div style="width: {rem_pct}%; background-color: #00CC96;"></div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.divider()

                        if cluster.skills:
                            tags = "".join([
                                               f"<span style='background:#f0f2f6; color:#333333; padding:2px 6px;margin:2px;border-radius:4px;font-size:0.8em;border:1px solid #ddd;display:inline-block'>{s}</span>"
                                               for s in cluster.skills])
                            st.markdown(f"**Skills:** {tags}", unsafe_allow_html=True)
                        else:
                            st.caption("Навыки не указаны")
