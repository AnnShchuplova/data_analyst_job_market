import os
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data(ttl=300)
def _load_data():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    processed_dir = os.path.join(project_root, "finaldata")
    if not os.path.exists(processed_dir):
        return None, None
    files = [f for f in os.listdir(processed_dir) if f.endswith(".csv")]
    if not files:
        return None, None
    latest_file = max(
        files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x))
    )
    file_path = os.path.join(processed_dir, latest_file)
    df = pd.read_csv(file_path, encoding="utf-8")
    return df, latest_file


def view_aggregation(service=None):
    st.markdown(
        "<h1 style='text-align: center;'>📊 АГРЕГАЦИЯ ВАКАНСИЙ</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Обзор рынка труда по собранным данным")

    try:
        df, filename = _load_data()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return

    if df is None:
        st.warning("📥 Данные не найдены. Поместите CSV в папку `finaldata/`.")
        return

    with st.container(border=True):
        st.subheader("⚙️ Параметры анализа")
        c1, c2, c3 = st.columns(3)
        with c1:
            regions = ["Все регионы", "Москва", "Санкт-Петербург", "Другие"]
            selected_region = st.selectbox("Регион", regions, key="agg_region")
        with c2:
            salary_range = st.slider(
                "Диапазон зарплат (тыс. руб.)", 0, 500, (0, 300), key="agg_salary"
            )
        with c3:
            experience = st.multiselect(
                "Опыт работы",
                ["Нет опыта", "1-3 года", "3-6 лет", "Более 6 лет"],
                default=["1-3 года", "3-6 лет"],
                key="agg_exp",
            )

    df_filtered = df.copy()

    if selected_region != "Все регионы" and "area_name" in df_filtered.columns:
        if selected_region == "Другие":
            df_filtered = df_filtered[
                ~df_filtered["area_name"].isin(["Москва", "Санкт-Петербург"])
            ]
        else:
            df_filtered = df_filtered[df_filtered["area_name"] == selected_region]

    if salary_range != (0, 500) and "salary_avg" in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered["salary_avg"].fillna(0) / 1000 >= salary_range[0])
            & (df_filtered["salary_avg"].fillna(999) / 1000 <= salary_range[1])
        ]

    if experience:
        exp_map = {
            "Нет опыта": "noExperience",
            "1-3 года": "between1And3",
            "3-6 лет": "between3And6",
            "Более 6 лет": "moreThan6",
        }
        exp_ids = [exp_map[e] for e in experience if e in exp_map]
        if exp_ids and "experience_id" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["experience_id"].isin(exp_ids)]

    df = df_filtered

    st.success(
        f"✅ Данные загружены: {filename} ({len(df)} записей после фильтрации)"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Обзор", "💰 Зарплаты", "🏙️ Регионы", "📋 Таблица"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                if "experience_name" in df.columns:
                    exp_counts = df["experience_name"].value_counts()
                    fig1 = px.pie(
                        values=exp_counts.values,
                        names=exp_counts.index,
                        title="Распределение по опыту работы",
                    )
                    st.plotly_chart(fig1, use_container_width=True)
        with col2:
            with st.container(border=True):
                if "salary_avg" in df.columns:
                    fig2 = px.histogram(
                        df,
                        x="salary_avg",
                        nbins=20,
                        title="Распределение зарплат",
                        labels={"salary_avg": "Средняя зарплата, руб."},
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.subheader("Анализ зарплат")
            if "salary_avg" in df.columns and "experience_name" in df.columns:
                fig3 = px.box(
                    df,
                    x="experience_name",
                    y="salary_avg",
                    title="Зарплаты по уровням опыта",
                    labels={
                        "salary_avg": "Зарплата, руб.",
                        "experience_name": "Опыт",
                    },
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Нужные столбцы отсутствуют в данных.")

    with tab3:
        with st.container(border=True):
            st.subheader("Географическое распределение")
            area_col = "area_name" if "area_name" in df.columns else "area"
            if area_col in df.columns:
                if area_col == "area_name":
                    region_counts = (
                        df["area_name"].fillna("Не указан").value_counts()
                    )
                else:
                    region_counts = (
                        df["area"]
                        .apply(
                            lambda x: x.get("name")
                            if isinstance(x, dict)
                            else "Не указан"
                        )
                        .value_counts()
                    )
                fig4 = px.bar(
                    x=region_counts.index[:10],
                    y=region_counts.values[:10],
                    title="Топ-10 регионов по количеству вакансий",
                    labels={"x": "Регион", "y": "Количество вакансий"},
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Информация о регионах отсутствует.")

    with tab4:
        with st.container(border=True):
            st.subheader("Таблица данных")
            cols = [
                c for c in ["name", "salary_avg", "experience_name"] if c in df.columns
            ]
            if cols:
                st.dataframe(df[cols].head(20), use_container_width=True)
            else:
                st.dataframe(df.head(20), use_container_width=True)
