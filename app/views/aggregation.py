import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_data():
    """Загрузка последнего датасета из finaldata/."""
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


def _parse_skills(skills_raw) -> list:
    """Извлечение списка навыков из строки."""
    if pd.isna(skills_raw):
        return []
    if isinstance(skills_raw, list):
        return [str(s).strip().lower() for s in skills_raw if str(s).strip()]
    if isinstance(skills_raw, str):
        text = skills_raw.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        skills = []
        for part in text.split(","):
            s = part.strip().strip("'\"").lower()
            if s:
                skills.append(s)
        return skills
    return []


def _get_unique_sorted(series):
    """Уникальные значения, отсортированные по частоте."""
    if series is None:
        return []
    return list(series.dropna().value_counts().index)


def _get_top_skills(df, n=30):
    """Топ-навыков по частоте."""
    all_skills = []
    if df is not None and "skills_list" in df.columns:
        for raw in df["skills_list"].dropna():
            all_skills.extend(_parse_skills(raw))
    return [s for s, _ in Counter(all_skills).most_common(n)]


def _format_salary(val):
    """Форматирование зарплаты для отображения."""
    if pd.isna(val) or val == 0:
        return "—"
    if val >= 1000000:
        return f"{val / 1000000:.1f}M ₽"
    if val >= 100000:
        return f"{val // 1000}K ₽"
    return f"{int(val):,} ₽".replace(",", " ")


def _salary_percentile(value, series):
    """Процентиль значения в серии."""
    if series is None or len(series) == 0:
        return 0
    return float((series < value).sum() / len(series) * 100)


def _make_filter_container(df_source, key_prefix):
    """Создаёт контейнер с локальными фильтрами и возвращает отфильтрованный df."""
    # Собираем опции из текущего подмножества данных
    local_regions = _get_unique_sorted(df_source.get("area_name"))
    local_roles = _get_unique_sorted(df_source.get("main_role_name"))
    local_exp = _get_unique_sorted(df_source.get("experience_name"))
    local_formats = _get_unique_sorted(df_source.get("work_format_name"))

    c1, c2, c3 = st.columns(3)
    result = df_source.copy()

    with c1:
        reg = st.selectbox(
            "Регион",
            ["Все регионы"] + local_regions,
            key=f"{key_prefix}_region",
        )
    with c2:
        exp = st.selectbox(
            "Опыт",
            ["Все уровни"] + local_exp,
            key=f"{key_prefix}_exp",
        )
    with c3:
        fmt = st.selectbox(
            "Формат",
            ["Все форматы"] + local_formats,
            key=f"{key_prefix}_format",
        )

    if reg != "Все регионы" and "area_name" in result.columns:
        result = result[result["area_name"] == reg]
    if exp != "Все уровни" and "experience_name" in result.columns:
        result = result[result["experience_name"] == exp]
    if fmt != "Все форматы" and "work_format_name" in result.columns:
        result = result[result["work_format_name"] == fmt]

    return result


# ──────────────────────────────────────────────
# Основной view
# ──────────────────────────────────────────────

def view_aggregation(service=None):
    st.markdown(
        "<h1 style='text-align: center;'>АГРЕГАЦИЯ ВАКАНСИЙ</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Интерактивный анализ рынка труда аналитиков данных")

    # ---- Загрузка данных ----
    try:
        df, filename = _load_data()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return

    if df is None:
        st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`.")
        return

    # ---- Подготовка опций фильтров ----
    regions = _get_unique_sorted(df.get("area_name"))
    roles = _get_unique_sorted(df.get("main_role_name"))
    experiences = _get_unique_sorted(df.get("experience_name"))
    schedules = _get_unique_sorted(df.get("schedule_name"))
    work_formats = _get_unique_sorted(df.get("work_format_name"))
    employments = _get_unique_sorted(df.get("employment_name"))
    all_skills = _get_top_skills(df, n=30)

    # =============================================
    # SIDEBAR — ГЛОБАЛЬНЫЕ ФИЛЬТРЫ
    # =============================================
    with st.sidebar:
        st.header("Фильтры")

        search_query = st.text_input(
            "Поиск по названию вакансии",
            placeholder="Например: Аналитик данных...",
            key="agg_search",
        )

        st.subheader("Регион")
        default_regions = regions[:3] if regions else []
        region_filter = st.multiselect(
            "Регион",
            regions if regions else ["Не указано"],
            default=default_regions,
            key="agg_region",
        )
        if st.checkbox("Выбрать все регионы", key="agg_region_all"):
            region_filter = regions

        st.subheader("Опыт работы")
        exp_filter = st.multiselect(
            "Опыт",
            experiences if experiences else ["Не указано"],
            default=experiences if experiences else [],
            key="agg_exp",
        )

        st.subheader("Зарплата")
        if "salary_avg" in df.columns:
            salary_min = float(df["salary_avg"].dropna().min())
            salary_max = float(df["salary_avg"].dropna().max())
            salary_range = st.slider(
                "Диапазон (тыс. ₽)",
                int(salary_min / 1000),
                min(int(salary_max / 1000), 500),
                (int(salary_min / 1000), min(int(salary_max / 1000), 500)),
                key="agg_salary",
            )
        else:
            salary_range = (0, 500)

        st.subheader("Должность")
        role_filter = st.multiselect(
            "Должность",
            roles if roles else ["Не указано"],
            key="agg_role",
        )

        st.subheader("Формат работы")
        format_filter = st.multiselect(
            "Формат",
            work_formats if work_formats else ["Не указано"],
            key="agg_format",
        )

        st.divider()
        st.caption(f"Файл: {filename}")

    # =============================================
    # ПРИМЕНЕНИЕ ГЛОБАЛЬНЫХ ФИЛЬТРОВ
    # =============================================
    df_f = df.copy()

    if search_query and "name" in df_f.columns:
        mask = df_f["name"].fillna("").str.lower().str.contains(
            search_query.lower(), na=False
        )
        df_f = df_f[mask]

    if region_filter and "area_name" in df_f.columns:
        df_f = df_f[df_f["area_name"].isin(region_filter)]

    if "salary_avg" in df_f.columns and salary_range != (0, 500):
        df_f = df_f[
            (df_f["salary_avg"].fillna(0) / 1000 >= salary_range[0])
            & (df_f["salary_avg"].fillna(999999) / 1000 <= salary_range[1])
        ]

    if exp_filter and "experience_name" in df_f.columns:
        df_f = df_f[df_f["experience_name"].isin(exp_filter)]

    if role_filter and "main_role_name" in df_f.columns:
        df_f = df_f[df_f["main_role_name"].isin(role_filter)]

    if format_filter and "work_format_name" in df_f.columns:
        df_f = df_f[df_f["work_format_name"].isin(format_filter)]

    n_total = len(df)
    n_filtered = len(df_f)
    n_with_salary = (
        int(df_f["salary_avg"].notna().sum()) if "salary_avg" in df_f.columns else 0
    )

    # =============================================
    # KPI КАРТОЧКИ
    # =============================================
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Вакансий (фильтр)",
            f"{n_filtered:,}".replace(",", " "),
            delta=f"из {n_total:,}".replace(",", " "),
        )

    with k2:
        if n_with_salary > 0:
            avg_sal = df_f["salary_avg"].mean()
            st.metric("Средняя ЗП", _format_salary(avg_sal))
        else:
            st.metric("Средняя ЗП", "—")

    with k3:
        if n_with_salary > 0:
            med_sal = df_f["salary_avg"].median()
            st.metric("Медианная ЗП", _format_salary(med_sal))
        else:
            st.metric("Медианная ЗП", "—")

    with k4:
        if n_filtered > 0:
            avg_skills = df_f["skills_list"].apply(
                lambda x: len(_parse_skills(x)) if pd.notna(x) else 0
            ).mean()
            st.metric("Ср. кол-во навыков", f"{avg_skills:.1f}")
        else:
            st.metric("Ср. кол-во навыков", "—")

    st.divider()

    if n_filtered == 0:
        st.warning(
            "Нет вакансий по выбранным фильтрам. Попробуйте изменить параметры."
        )
        return

    # =============================================
    # ВКЛАДКИ
    # =============================================
    tab_overview, tab_salary, tab_skills, tab_geo, tab_profile, tab_table = st.tabs(
        [
            "📈 Обзор",
            "💰 Зарплаты",
            "🧠 Навыки",
            "🗺️ География",
            "👤 Мой профиль",
            "📋 Таблица",
        ]
    )

    # =============================================
    # TAB 1: ОБЗОР
    # =============================================
    with tab_overview:
        st.caption("Общая картина рынка: распределение по опыту, зарплатам и популярным должностям.")

        with st.container(border=True):
            st.subheader("⚙️ Параметры обзора")
            ov_role = st.selectbox(
                "Должность",
                ["Все должности"] + _get_unique_sorted(df_f.get("main_role_name")),
                key="ov_role",
            )
            ov_exp = st.multiselect(
                "Опыт",
                _get_unique_sorted(df_f.get("experience_name")),
                default=_get_unique_sorted(df_f.get("experience_name")),
                key="ov_exp",
            )

        df_ov = df_f.copy()
        if ov_role != "Все должности" and "main_role_name" in df_ov.columns:
            df_ov = df_ov[df_ov["main_role_name"] == ov_role]
        if ov_exp and "experience_name" in df_ov.columns:
            df_ov = df_ov[df_ov["experience_name"].isin(ov_exp)]

        if len(df_ov) == 0:
            st.warning("Нет данных по выбранным параметрам.")
        else:
            st.success(f"Показано: {len(df_ov)} вакансий")

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Распределение по опыту")
                if "experience_name" in df_ov.columns:
                    exp_counts = df_ov["experience_name"].value_counts()
                    fig = px.pie(
                        values=exp_counts.values,
                        names=exp_counts.index,
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Распределение зарплат")
                if "salary_avg" in df_ov.columns:
                    salary_data = df_ov["salary_avg"].dropna()
                    salary_data = salary_data[salary_data > 0]
                    if len(salary_data) > 0:
                        fig = px.histogram(
                            salary_data,
                            x="salary_avg",
                            nbins=30,
                            color_discrete_sequence=["#4C72B0"],
                            marginal="violin",
                            title="",
                        )
                        mean_val = salary_data.mean()
                        med_val = salary_data.median()
                        fig.add_vline(
                            x=mean_val,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Среднее: {mean_val / 1000:.0f}K",
                            annotation_position="top left",
                        )
                        fig.add_vline(
                            x=med_val,
                            line_dash="dot",
                            line_color="orange",
                            annotation_text=f"Медиана: {med_val / 1000:.0f}K",
                            annotation_position="top right",
                        )
                        fig.update_layout(
                            height=400,
                            xaxis_title="Зарплата, ₽",
                            yaxis_title="Количество",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.subheader("Топ-10 должностей по количеству вакансий")
            if "main_role_name" in df_ov.columns:
                role_counts = df_ov["main_role_name"].value_counts().head(10)
                fig = px.bar(
                    x=role_counts.values,
                    y=role_counts.index,
                    orientation="h",
                    color=role_counts.values,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    height=350,
                    xaxis_title="Количество вакансий",
                    yaxis_title="",
                    showlegend=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =============================================
    # TAB 2: ЗАРПЛАТЫ
    # =============================================
    with tab_salary:
        st.caption("Детальный разбор зарплат: по опыту, должностям и формату работы.")

        with st.container(border=True):
            st.subheader("⚙️ Параметры зарплат")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                sal_role = st.selectbox(
                    "Должность",
                    ["Все должности"] + _get_unique_sorted(df_f.get("main_role_name")),
                    key="sal_role",
                )
            with sc2:
                sal_exp = st.selectbox(
                    "Опыт",
                    ["Все уровни"] + _get_unique_sorted(df_f.get("experience_name")),
                    key="sal_exp",
                )
            with sc3:
                sal_format = st.selectbox(
                    "Формат работы",
                    ["Все форматы"] + _get_unique_sorted(df_f.get("work_format_name")),
                    key="sal_format",
                )

        df_sal = df_f.copy()
        if sal_role != "Все должности" and "main_role_name" in df_sal.columns:
            df_sal = df_sal[df_sal["main_role_name"] == sal_role]
        if sal_exp != "Все уровни" and "experience_name" in df_sal.columns:
            df_sal = df_sal[df_sal["experience_name"] == sal_exp]
        if sal_format != "Все форматы" and "work_format_name" in df_sal.columns:
            df_sal = df_sal[df_sal["work_format_name"] == sal_format]

        if len(df_sal) == 0:
            st.warning("Нет данных по выбранным параметрам.")
        else:
            st.success(f"Показано: {len(df_sal)} вакансий")

            df_salary = df_sal.dropna(subset=["salary_avg"])
            df_salary = df_salary[df_salary["salary_avg"] > 0]

            # Box plot по опыту
            st.subheader("Зарплаты по уровню опыта")
            if "experience_name" in df_salary.columns and len(df_salary) > 0:
                fig = px.box(
                    df_salary,
                    x="experience_name",
                    y="salary_avg",
                    color="experience_name",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    height=450,
                    xaxis_title="Опыт работы",
                    yaxis_title="Зарплата, ₽",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Зарплаты по должностям (топ-8)")
                if "main_role_name" in df_salary.columns:
                    top_roles = df_salary["main_role_name"].value_counts().head(8).index
                    df_roles = df_salary[df_salary["main_role_name"].isin(top_roles)]

                    fig = px.box(
                        df_roles,
                        y="main_role_name",
                        x="salary_avg",
                        color="main_role_name",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    )
                    fig.update_layout(
                        height=450,
                        xaxis_title="Зарплата, ₽",
                        yaxis_title="",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Статистика зарплат")
                if "experience_name" in df_salary.columns:
                    stats = (
                        df_salary.groupby("experience_name")["salary_avg"]
                        .agg(["count", "mean", "median", "min", "max"])
                        .round(0)
                    )
                    stats.columns = ["Кол-во", "Средняя", "Медиана", "Мин", "Макс"]
                    stats["Средняя"] = stats["Средняя"].apply(_format_salary)
                    stats["Медиана"] = stats["Медиана"].apply(_format_salary)
                    stats["Мин"] = stats["Мин"].apply(_format_salary)
                    stats["Макс"] = stats["Макс"].apply(_format_salary)
                    st.dataframe(stats, use_container_width=True, height=250)

                st.subheader("Удалёнка vs Офис")
                if "work_format_name" in df_salary.columns:
                    fmt_stats = (
                        df_salary.groupby("work_format_name")["salary_avg"]
                        .agg(["count", "mean", "median"])
                        .round(0)
                    )
                    fmt_stats.columns = ["Кол-во", "Средняя", "Медиана"]
                    fmt_stats["Средняя"] = fmt_stats["Средняя"].apply(_format_salary)
                    fmt_stats["Медиана"] = fmt_stats["Медиана"].apply(_format_salary)
                    st.dataframe(fmt_stats, use_container_width=True, height=200)

    # =============================================
    # TAB 3: НАВЫКИ
    # =============================================
    with tab_skills:
        st.caption("Анализ востребованности навыков и их влияния на зарплату. Выберите навык для детального сравнения.")

        with st.container(border=True):
            st.subheader("⚙️ Параметры навыков")
            sk1, sk2 = st.columns(2)
            with sk1:
                sk_exp = st.selectbox(
                    "Опыт",
                    ["Все уровни"] + _get_unique_sorted(df_f.get("experience_name")),
                    key="sk_exp",
                )
            with sk2:
                sk_role = st.selectbox(
                    "Должность",
                    ["Все должности"] + _get_unique_sorted(df_f.get("main_role_name")),
                    key="sk_role",
                )

        df_sk = df_f.copy()
        if sk_exp != "Все уровни" and "experience_name" in df_sk.columns:
            df_sk = df_sk[df_sk["experience_name"] == sk_exp]
        if sk_role != "Все должности" and "main_role_name" in df_sk.columns:
            df_sk = df_sk[df_sk["main_role_name"] == sk_role]

        if len(df_sk) == 0:
            st.warning("Нет данных по выбранным параметрам.")
        else:
            st.success(f"Показано: {len(df_sk)} вакансий")

            # Детальный анализ конкретного навыка
            st.subheader("Детальный анализ навыка")
            tab_skills_list = _get_top_skills(df_sk, n=30)
            skill_choice = st.selectbox(
                "Выберите навык для анализа",
                [""] + tab_skills_list,
                format_func=lambda x: "Выберите навык..." if x == "" else x.capitalize(),
                key="agg_skill_detail",
            )

            if skill_choice:
                col_a, col_b, col_c = st.columns(3)

                if "salary_avg" in df_sk.columns and "skills_list" in df_sk.columns:
                    df_sal = df_sk.dropna(subset=["salary_avg"])
                    df_sal = df_sal[df_sal["salary_avg"] > 0]

                    has_skill = df_sal[
                        df_sal["skills_list"].apply(
                            lambda x: skill_choice.lower() in _parse_skills(x)
                        )
                    ]
                    no_skill = df_sal[
                        ~df_sal["skills_list"].apply(
                            lambda x: skill_choice.lower() in _parse_skills(x)
                        )
                    ]

                    pct = len(has_skill) / len(df_sal) * 100 if len(df_sal) > 0 else 0
                    avg_with = has_skill["salary_avg"].mean() if len(has_skill) > 0 else 0
                    avg_without = no_skill["salary_avg"].mean() if len(no_skill) > 0 else 0
                    diff_pct = (
                        (avg_with - avg_without) / avg_without * 100
                        if avg_without > 0
                        else 0
                    )

                    with col_a:
                        st.metric(
                            "Встречаемость",
                            f"{pct:.1f}%",
                            delta=f"{len(has_skill)} из {len(df_sal)}",
                        )
                    with col_b:
                        st.metric(
                            "ЗП с навыком",
                            _format_salary(avg_with) if avg_with > 0 else "—",
                        )
                    with col_c:
                        delta_label = (
                            f"+{diff_pct:.1f}%" if diff_pct > 0 else f"{diff_pct:.1f}%"
                        )
                        st.metric(
                            "ЗП без навыка",
                            _format_salary(avg_without) if avg_without > 0 else "—",
                            delta=delta_label,
                            delta_color="normal" if diff_pct <= 0 else "inverse",
                        )

                    c1, c2 = st.columns(2)
                    with c1:
                        if len(has_skill) > 2:
                            fig = px.histogram(
                                has_skill,
                                x="salary_avg",
                                nbins=20,
                                color_discrete_sequence=["#2ecc71"],
                            )
                            fig.update_layout(
                                height=300,
                                showlegend=False,
                                xaxis_title="Зарплата, ₽",
                                yaxis_title="Количество",
                                title=f"С навыком «{skill_choice}»",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Слишком мало данных с этим навыком")

                    with c2:
                        if len(no_skill) > 2:
                            fig = px.histogram(
                                no_skill,
                                x="salary_avg",
                                nbins=20,
                                color_discrete_sequence=["#e74c3c"],
                            )
                            fig.update_layout(
                                height=300,
                                showlegend=False,
                                xaxis_title="Зарплата, ₽",
                                yaxis_title="Количество",
                                title=f"Без навыка «{skill_choice}»",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Слишком мало данных без этого навыка")
                else:
                    st.warning("Нет данных о зарплатах для анализа навыков")

            st.divider()

            # Топ-20 навыков по встречаемости
            st.subheader("Топ-20 навыков по встречаемости")
            skills_counter = Counter()
            if "skills_list" in df_sk.columns:
                for raw in df_sk["skills_list"].dropna():
                    skills_counter.update(_parse_skills(raw))

            top_skills = skills_counter.most_common(20)
            if top_skills:
                skill_names = [s[0].capitalize() for s in top_skills]
                skill_counts = [s[1] for s in top_skills]

                fig = px.bar(
                    x=skill_counts,
                    y=skill_names,
                    orientation="h",
                    color=skill_counts,
                    color_continuous_scale="Teal",
                )
                fig.update_layout(
                    height=500,
                    xaxis_title="Встречаемость в вакансиях",
                    yaxis_title="",
                    showlegend=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Топ-15 навыков по средней зарплате
            st.subheader("Топ-15 навыков по средней зарплате")
            if "salary_avg" in df_sk.columns and "skills_list" in df_sk.columns:
                df_sal = df_sk.dropna(subset=["salary_avg"])
                df_sal = df_sal[df_sal["salary_avg"] > 0]

                skill_salary = {}
                skill_count = {}
                for _, row in df_sal.iterrows():
                    sks = _parse_skills(row["skills_list"])
                    for sk in sks:
                        skill_salary[sk] = skill_salary.get(sk, []) + [row["salary_avg"]]
                        skill_count[sk] = skill_count.get(sk, 0) + 1

                filtered = {
                    k: np.mean(v)
                    for k, v in skill_salary.items()
                    if skill_count.get(k, 0) >= 10
                }
                sorted_skills = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[
                    :15
                ]

                if sorted_skills:
                    fig = px.bar(
                        x=[s[1] for s in sorted_skills],
                        y=[s[0].capitalize() for s in sorted_skills],
                        orientation="h",
                        color=[s[1] for s in sorted_skills],
                        color_continuous_scale="YlOrRd",
                    )
                    fig.update_layout(
                        height=500,
                        xaxis_title="Средняя зарплата, ₽",
                        yaxis_title="",
                        showlegend=False,
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Недостаточно данных (нужно 10+ вакансий с навыком)")

    # =============================================
    # TAB 4: ГЕОГРАФИЯ
    # =============================================
    with tab_geo:
        st.caption("Распределение вакансий и зарплат по регионам. Сравните средние и медианные значения.")

        with st.container(border=True):
            st.subheader("⚙️ Параметры географии")
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                geo_region = st.selectbox(
                    "Город / регион",
                    ["Все регионы"] + _get_unique_sorted(df_f.get("area_name")),
                    key="geo_region",
                )
            with gc2:
                geo_exp = st.selectbox(
                    "Опыт",
                    ["Все уровни"] + _get_unique_sorted(df_f.get("experience_name")),
                    key="geo_exp",
                )
            with gc3:
                geo_role = st.selectbox(
                    "Должность",
                    ["Все должности"] + _get_unique_sorted(df_f.get("main_role_name")),
                    key="geo_role",
                )

        df_geo = df_f.copy()
        if geo_region != "Все регионы" and "area_name" in df_geo.columns:
            df_geo = df_geo[df_geo["area_name"] == geo_region]
        if geo_exp != "Все уровни" and "experience_name" in df_geo.columns:
            df_geo = df_geo[df_geo["experience_name"] == geo_exp]
        if geo_role != "Все должности" and "main_role_name" in df_geo.columns:
            df_geo = df_geo[df_geo["main_role_name"] == geo_role]

        if len(df_geo) == 0:
            st.warning("Нет данных по выбранным параметрам.")
        else:
            st.success(f"Показано: {len(df_geo)} вакансий")

            # Если выбран конкретный регион — показываем детальную аналитику по нему
            if geo_region != "Все регионы":
                st.markdown(f"### Детальная аналитика: {geo_region}")

                if "salary_avg" in df_geo.columns:
                    df_gs = df_geo.dropna(subset=["salary_avg"])
                    df_gs = df_gs[df_gs["salary_avg"] > 0]

                    gc1, gc2, gc3, gc4 = st.columns(4)
                    with gc1:
                        st.metric("Всего вакансий", len(df_geo))
                    with gc2:
                        st.metric(
                            "Средняя ЗП",
                            _format_salary(df_gs["salary_avg"].mean())
                            if len(df_gs) > 0
                            else "—",
                        )
                    with gc3:
                        st.metric(
                            "Медианная ЗП",
                            _format_salary(df_gs["salary_avg"].median())
                            if len(df_gs) > 0
                            else "—",
                        )
                    with gc4:
                        if len(df_gs) > 0:
                            st.metric(
                                "Диапазон ЗП",
                                f"{_format_salary(df_gs['salary_avg'].min())} — "
                                f"{_format_salary(df_gs['salary_avg'].max())}",
                            )

                    # Распределение зарплат в регионе
                    if len(df_gs) > 0:
                        st.subheader("Распределение зарплат")
                        fig = px.histogram(
                            df_gs,
                            x="salary_avg",
                            nbins=25,
                            color_discrete_sequence=["#4C72B0"],
                        )
                        mean_val = df_gs["salary_avg"].mean()
                        fig.add_vline(
                            x=mean_val,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Среднее: {mean_val / 1000:.0f}K",
                        )
                        fig.update_layout(
                            height=350,
                            xaxis_title="Зарплата, ₽",
                            yaxis_title="Количество",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Зарплаты по опыту в регионе
                    if "experience_name" in df_gs.columns and len(df_gs) > 0:
                        st.subheader("Зарплаты по опыту")
                        fig = px.box(
                            df_gs,
                            x="experience_name",
                            y="salary_avg",
                            color="experience_name",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        fig.update_layout(
                            height=400,
                            xaxis_title="Опыт",
                            yaxis_title="Зарплата, ₽",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Зарплаты по должностям в регионе
                    if "main_role_name" in df_gs.columns and len(df_gs) > 0:
                        st.subheader("Зарплаты по должностям")
                        top_geo_roles = (
                            df_gs["main_role_name"].value_counts().head(8).index
                        )
                        df_gr = df_gs[df_gs["main_role_name"].isin(top_geo_roles)]
                        fig = px.box(
                            df_gr,
                            y="main_role_name",
                            x="salary_avg",
                            color="main_role_name",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                        )
                        fig.update_layout(
                            height=max(350, len(top_geo_roles) * 50),
                            xaxis_title="Зарплата, ₽",
                            yaxis_title="",
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Топ навыков в регионе
                    if "skills_list" in df_geo.columns:
                        st.subheader("Топ-15 навыков в регионе")
                        geo_skills_counter = Counter()
                        for raw in df_geo["skills_list"].dropna():
                            geo_skills_counter.update(_parse_skills(raw))
                        geo_top = geo_skills_counter.most_common(15)
                        if geo_top:
                            fig = px.bar(
                                x=[s[1] for s in geo_top],
                                y=[s[0].capitalize() for s in geo_top],
                                orientation="h",
                                color=[s[1] for s in geo_top],
                                color_continuous_scale="Viridis",
                            )
                            fig.update_layout(
                                height=450,
                                xaxis_title="Встречаемость",
                                yaxis_title="",
                                showlegend=False,
                                yaxis=dict(autorange="reversed"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # Форматы работы в регионе
                if "work_format_name" in df_geo.columns:
                    st.subheader("Форматы работы")
                    fmt_counts = df_geo["work_format_name"].value_counts()
                    fig = px.pie(
                        values=fmt_counts.values,
                        names=fmt_counts.index,
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Все регионы — показываем сравнительные графики
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("Топ-15 регионов по количеству")
                    if "area_name" in df_geo.columns:
                        region_counts = (
                            df_geo["area_name"].fillna("Не указан").value_counts().head(15)
                        )
                        fig = px.bar(
                            x=region_counts.values,
                            y=region_counts.index,
                            orientation="h",
                            color=region_counts.values,
                            color_continuous_scale="Ice",
                        )
                        fig.update_layout(
                            height=500,
                            xaxis_title="Количество вакансий",
                            showlegend=False,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.subheader("Топ-10 регионов по средней ЗП")
                    if "salary_avg" in df_geo.columns and "area_name" in df_geo.columns:
                        df_sal = df_geo.dropna(subset=["salary_avg"])
                        df_sal = df_sal[df_sal["salary_avg"] > 0]
                        region_counts_sal = df_sal["area_name"].value_counts()
                        big_regions = region_counts_sal[region_counts_sal >= 5].index
                        df_big = df_sal[df_sal["area_name"].isin(big_regions)]
                        region_salary = (
                            df_big.groupby("area_name")["salary_avg"]
                            .agg(["mean", "median", "count"])
                            .sort_values("mean", ascending=False)
                            .head(10)
                        )

                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                name="Средняя ЗП",
                                x=region_salary["mean"],
                                y=region_salary.index,
                                orientation="h",
                                marker_color="#4C72B0",
                            )
                        )
                        fig.add_trace(
                            go.Bar(
                                name="Медианная ЗП",
                                x=region_salary["median"],
                                y=region_salary.index,
                                orientation="h",
                                marker_color="#55A868",
                            )
                        )
                        fig.update_layout(
                            barmode="group",
                            height=500,
                            xaxis_title="Зарплата, ₽",
                            showlegend=True,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.subheader("Детальная таблица по регионам")
                if "salary_avg" in df_geo.columns and "area_name" in df_geo.columns:
                    df_sal = df_geo.dropna(subset=["salary_avg"])
                    df_sal = df_sal[df_sal["salary_avg"] > 0]
                    region_stats = (
                        df_sal.groupby("area_name")["salary_avg"]
                        .agg(["count", "mean", "median", "min", "max"])
                        .round(0)
                        .sort_values("count", ascending=False)
                        .head(15)
                    )
                    region_stats.columns = ["Вакансий", "Средняя", "Медиана", "Мин", "Макс"]
                    region_stats["Средняя"] = region_stats["Средняя"].apply(_format_salary)
                    region_stats["Медиана"] = region_stats["Медиана"].apply(_format_salary)
                    region_stats["Мин"] = region_stats["Мин"].apply(_format_salary)
                    region_stats["Макс"] = region_stats["Макс"].apply(_format_salary)
                    st.dataframe(region_stats, use_container_width=True, height=400)

    # =============================================
    # TAB 5: МОЙ ПРОФИЛЬ — персонализированный анализ
    # =============================================
    with tab_profile:
        st.markdown(
            "<div style='"
            "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
            "padding: 24px; border-radius: 12px; margin-bottom: 20px;"
            "'>"
            "<h2 style='color: #fff; margin: 0;'>👤 Персональная аналитика</h2>"
            "<p style='color: #e0e0ff; margin: 6px 0 0 0;'>"
            "Введите свои данные, чтобы узнать, как ваша зарплата соотносится с рынком, "
            "какие навыки стоит освоить и насколько вы входите в рынок.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        pc1, pc2 = st.columns(2)

        with pc1:
            my_salary = st.number_input(
                "Ваша текущая зарплата (₽)",
                min_value=0,
                max_value=1000000,
                value=0,
                step=5000,
                key="profile_salary",
                placeholder="Введите вашу зарплату",
            )
            my_experience = st.selectbox(
                "Ваш опыт",
                [""] + (experiences if experiences else []),
                format_func=lambda x: "Выберите опыт..." if x == "" else x,
                key="profile_exp",
            )
            my_region = st.selectbox(
                "Ваш регион",
                [""] + (regions if regions else []),
                format_func=lambda x: "Выберите регион..." if x == "" else x,
                key="profile_region",
            )

        with pc2:
            my_role = st.selectbox(
                "Ваша должность",
                [""] + (roles if roles else []),
                format_func=lambda x: "Выберите должность..." if x == "" else x,
                key="profile_role",
            )
            my_skills = st.multiselect(
                "Ваши навыки",
                all_skills if all_skills else [],
                key="profile_skills",
            )
            my_format = st.selectbox(
                "Ваш формат работы",
                [""] + (work_formats if work_formats else []),
                format_func=lambda x: "Выберите формат..." if x == "" else x,
                key="profile_format",
            )

        if my_salary > 0:
            st.divider()

            mc1, mc2, mc3, mc4 = st.columns(4)

            df_all_salary = df_f.dropna(subset=["salary_avg"])
            df_all_salary = df_all_salary[df_all_salary["salary_avg"] > 0]
            all_salaries = df_all_salary["salary_avg"]

            percentile = _salary_percentile(my_salary, all_salaries)
            avg_market = all_salaries.mean()
            diff_from_avg = my_salary - avg_market
            diff_pct = (diff_from_avg / avg_market * 100) if avg_market > 0 else 0

            with mc1:
                st.metric(
                    "Ваш процентиль",
                    f"{percentile:.0f}%",
                    help="Процент людей, которые зарабатывают меньше вас",
                )
            with mc2:
                st.metric("Средняя по рынку", _format_salary(avg_market))
            with mc3:
                st.metric(
                    "Разница со средней",
                    f"{_format_salary(abs(diff_from_avg))}",
                    delta=f"{diff_pct:+.1f}%",
                    delta_color="normal" if diff_pct <= 0 else "inverse",
                )
            with mc4:
                p25 = all_salaries.quantile(0.25)
                p75 = all_salaries.quantile(0.75)
                st.metric(
                    "Рыночный коридор",
                    f"{_format_salary(p25)} — {_format_salary(p75)}",
                )

            st.subheader("Ваша позиция на рынке")
            fig = px.histogram(
                all_salaries,
                x="salary_avg",
                nbins=40,
                color_discrete_sequence=["#4C72B0"],
            )
            fig.add_vline(
                x=my_salary,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"Вы: {_format_salary(my_salary)}",
                annotation_position="top left",
                annotation_font=dict(size=14, color="red"),
            )
            fig.add_vline(
                x=avg_market,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Среднее: {_format_salary(avg_market)}",
                annotation_position="top right",
            )
            fig.add_vline(
                x=all_salaries.median(),
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Медиана: {_format_salary(all_salaries.median())}",
                annotation_position="bottom right",
            )
            fig.update_layout(
                height=400,
                xaxis_title="Зарплата, ₽",
                yaxis_title="Количество",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Сравнение по опыту
            if my_experience and "experience_name" in df_f.columns:
                df_exp = df_all_salary[
                    df_all_salary["experience_name"] == my_experience
                ]
                if len(df_exp) > 0:
                    st.subheader(f"Сравнение с коллегами ({my_experience})")
                    ec1, ec2, ec3 = st.columns(3)

                    exp_avg = df_exp["salary_avg"].mean()
                    exp_med = df_exp["salary_avg"].median()
                    exp_p = _salary_percentile(my_salary, df_exp["salary_avg"])

                    with ec1:
                        st.metric("Средняя по опыту", _format_salary(exp_avg))
                    with ec2:
                        st.metric("Медиана по опыту", _format_salary(exp_med))
                    with ec3:
                        st.metric("Ваш процентиль среди коллег", f"{exp_p:.0f}%")

                    fig = px.histogram(
                        df_exp,
                        x="salary_avg",
                        nbins=25,
                        color_discrete_sequence=["#8FB0D4"],
                    )
                    fig.add_vline(
                        x=my_salary,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"Вы: {_format_salary(my_salary)}",
                        annotation_position="top left",
                        annotation_font=dict(size=13, color="red"),
                    )
                    fig.update_layout(
                        height=350,
                        xaxis_title="Зарплата, ₽",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Сравнение по региону
            if my_region and "area_name" in df_f.columns:
                df_reg = df_all_salary[
                    df_all_salary["area_name"] == my_region
                ]
                if len(df_reg) > 5:
                    st.subheader(f"Сравнение по региону: {my_region}")
                    rc1, rc2, rc3 = st.columns(3)

                    reg_avg = df_reg["salary_avg"].mean()
                    reg_med = df_reg["salary_avg"].median()
                    reg_p = _salary_percentile(my_salary, df_reg["salary_avg"])

                    with rc1:
                        st.metric("Средняя по региону", _format_salary(reg_avg))
                    with rc2:
                        st.metric("Медиана по региону", _format_salary(reg_med))
                    with rc3:
                        st.metric("Ваш процентиль в регионе", f"{reg_p:.0f}%")

            # Сравнение по навыкам
            if my_skills:
                st.subheader("Ваши навыки и их влияние на зарплату")

                skill_impact_data = []
                for sk in my_skills:
                    if "salary_avg" in df_f.columns and "skills_list" in df_f.columns:
                        has_sk = df_all_salary[
                            df_all_salary["skills_list"].apply(
                                lambda x: sk.lower() in _parse_skills(x)
                            )
                        ]
                        no_sk = df_all_salary[
                            ~df_all_salary["skills_list"].apply(
                                lambda x: sk.lower() in _parse_skills(x)
                            )
                        ]
                        avg_has = (
                            has_sk["salary_avg"].mean() if len(has_sk) > 0 else 0
                        )
                        avg_no = (
                            no_sk["salary_avg"].mean() if len(no_sk) > 0 else 0
                        )
                        diff = avg_has - avg_no
                        skill_impact_data.append(
                            {
                                "Навык": sk.capitalize(),
                                "ЗП с навыком": _format_salary(avg_has),
                                "ЗП без навыка": _format_salary(avg_no),
                                "Разница": (
                                    f"{diff:+,.0f} ₽".replace(",", " ")
                                    if avg_no > 0
                                    else "—"
                                ),
                                "Встречаемость": f"{len(has_sk)}",
                            }
                        )

                if skill_impact_data:
                    df_impact = pd.DataFrame(skill_impact_data)
                    st.dataframe(df_impact, use_container_width=True, hide_index=True)

            # Рекомендации
            if my_skills and "salary_avg" in df_f.columns:
                st.subheader("Что стоит выучить? (навыки, которые поднимают ЗП)")

                skills_counter_global = Counter()
                if "skills_list" in df_f.columns:
                    for raw in df_f["skills_list"].dropna():
                        skills_counter_global.update(_parse_skills(raw))

                my_skills_lower = [s.lower() for s in my_skills]
                candidates = {}
                for sk, cnt in skills_counter_global.items():
                    if sk in my_skills_lower:
                        continue
                    if cnt < 15:
                        continue
                    has_sk = df_all_salary[
                        df_all_salary["skills_list"].apply(
                            lambda x: sk in _parse_skills(x)
                        )
                    ]
                    if len(has_sk) < 10:
                        continue
                    avg_has = has_sk["salary_avg"].mean()
                    no_sk = df_all_salary[
                        ~df_all_salary["skills_list"].apply(
                            lambda x: sk in _parse_skills(x)
                        )
                    ]
                    avg_no = no_sk["salary_avg"].mean()
                    boost = avg_has - avg_no
                    if boost > 0:
                        candidates[sk] = {
                            "bonus": boost,
                            "avg_with": avg_has,
                            "count": len(has_sk),
                        }

                top_candidates = sorted(
                    candidates.items(), key=lambda x: x[1]["bonus"], reverse=False
                )[:15]

                if top_candidates:
                    fig = px.bar(
                        x=[c[1]["bonus"] for c in top_candidates],
                        y=[c[0].capitalize() for c in top_candidates],
                        orientation="h",
                        color=[c[1]["bonus"] for c in top_candidates],
                        color_continuous_scale="RdYlGn",
                    )
                    fig.update_layout(
                        height=max(350, len(top_candidates) * 30),
                        xaxis_title="Прибавка к зарплате, ₽",
                        yaxis_title="",
                        showlegend=False,
                    )
                    fig.update_traces(
                        hovertemplate="%{y}: +%{x:,.0f} ₽<extra></extra>"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Недостаточно данных для рекомендаций")

        else:
            st.info(
                "Введите вашу зарплату выше, чтобы получить персонализированный анализ"
            )

    # =============================================
    # TAB 6: ТАБЛИЦА
    # =============================================
    with tab_table:
        st.caption("Исходные данные с возможностью выбора колонок и фильтрации.")

        with st.container(border=True):
            st.subheader("⚙️ Параметры таблицы")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                tbl_search = st.text_input(
                    "Поиск по названию",
                    placeholder="Фильтр по названию...",
                    key="tbl_search",
                )
            with tc2:
                tbl_sort = st.selectbox(
                    "Сортировка",
                    ["Без сортировки", "ЗП по убыванию", "ЗП по возрастанию", "По названию"],
                    key="tbl_sort",
                )
            with tc3:
                tbl_limit = st.slider(
                    "Кол-во записей",
                    10,
                    500,
                    100,
                    step=10,
                    key="tbl_limit",
                )

        df_tbl = df_f.copy()

        if tbl_search and "name" in df_tbl.columns:
            mask = df_tbl["name"].fillna("").str.lower().str.contains(
                tbl_search.lower(), na=False
            )
            df_tbl = df_tbl[mask]

        if tbl_sort == "ЗП по убыванию" and "salary_avg" in df_tbl.columns:
            df_tbl = df_tbl.sort_values("salary_avg", ascending=False)
        elif tbl_sort == "ЗП по возрастанию" and "salary_avg" in df_tbl.columns:
            df_tbl = df_tbl.sort_values("salary_avg", ascending=True)
        elif tbl_sort == "По названию" and "name" in df_tbl.columns:
            df_tbl = df_tbl.sort_values("name")

        st.subheader(
            f"Таблица данных ({len(df_tbl):,} записей)".replace(",", " ")
        )

        all_cols = list(df_tbl.columns)
        default_display = [
            c
            for c in [
                "name",
                "salary_avg",
                "area_name",
                "experience_name",
                "main_role_name",
                "work_format_name",
                "skills_list",
            ]
            if c in df_tbl.columns
        ]
        if not default_display:
            default_display = all_cols[:7]

        selected_cols = st.multiselect(
            "Выберите колонки для отображения",
            all_cols,
            default=default_display,
            key="agg_table_cols",
        )

        if selected_cols:
            st.dataframe(
                df_tbl[selected_cols].head(tbl_limit),
                use_container_width=True,
                height=500,
            )
            if len(df_tbl) > tbl_limit:
                st.caption(
                    f"Показаны первые {tbl_limit} из {len(df_tbl):,} записей".replace(
                        ",", " "
                    )
                )
        else:
            st.dataframe(df_tbl.head(tbl_limit), use_container_width=True)
