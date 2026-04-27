import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(ttl=300)
def _load_data():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_dir = os.path.join(project_root, "finaldata")
    if not os.path.exists(data_dir):
        return None, None
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        return None, None
    latest_file = max(
        files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x))
    )
    file_path = os.path.join(data_dir, latest_file)
    df = pd.read_csv(file_path, encoding="utf-8")
    return df, latest_file


def _project_root():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def _get_unique_sorted(series):
    """Уникальные значения, отсортированные по частоте."""
    if series is None:
        return []
    return list(series.dropna().value_counts().index)


def _parse_datetime(series):
    """Безопасный парсинг дат — поддерживает разные форматы."""
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    # Пробуем разные форматы
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    # Убираем таймзону для единообразия
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def _build_vacancy_timeline(df, region_filter, exp_filter, freq="W"):
    """
    Строит временной ряд количества вакансий по датам.

    Args:
        df: исходный DataFrame
        region_filter: список регионов или None
        exp_filter: список опыта или None
        freq: частота агрегации (D, W, M)

    Returns:
        DataFrame с колонками date и count, или None
    """
    result = df.copy()

    # Фильтруем по региону
    if region_filter and "area_name" in result.columns:
        result = result[result["area_name"].isin(region_filter)]

    # Фильтруем по опыту
    if exp_filter and "experience_name" in result.columns:
        result = result[result["experience_name"].isin(exp_filter)]

    if len(result) == 0:
        return None

    # Парсим даты
    if "published_at" not in result.columns:
        return None

    result["_date"] = _parse_datetime(result["published_at"])
    result = result.dropna(subset=["_date"])

    if len(result) == 0:
        return None

    # Агрегируем по частоте
    result = result.set_index("_date")

    freq_map = {"D": "День", "W": "Неделя", "ME": "Месяц"}
    freq_label = freq_map.get(freq, freq)

    ts = result.resample(freq).size().reset_index()
    ts.columns = ["date", "count"]

    return ts, freq_label


def _build_salary_timeline(df, region_filter, exp_filter, freq="W"):
    """
    Строит временной ряд средней зарплаты по датам.

    Returns:
        DataFrame с колонками date и avg_salary, или None
    """
    result = df.copy()

    if region_filter and "area_name" in result.columns:
        result = result[result["area_name"].isin(region_filter)]

    if exp_filter and "experience_name" in result.columns:
        result = result[result["experience_name"].isin(exp_filter)]

    if len(result) == 0:
        return None

    if "published_at" not in result.columns or "salary_avg" not in result.columns:
        return None

    result["_date"] = _parse_datetime(result["published_at"])
    result = result.dropna(subset=["_date", "salary_avg"])
    result = result[result["salary_avg"] > 0]

    if len(result) == 0:
        return None

    result = result.set_index("_date")
    ts = result.resample(freq)["salary_avg"].mean().reset_index()
    ts.columns = ["date", "avg_salary"]

    return ts


# ──────────────────────────────────────────────
# Основной view
# ──────────────────────────────────────────────

def view_arima(service=None):
    st.markdown(
        "<h1 style='text-align: center;'>📈 АНАЛИЗ ТРЕНДОВ (SARIMA)</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Прогнозирование количества вакансий во временем методом SARIMA "
        "и анализ динамики рынка."
    )

    project_root = _project_root()
    arima_metrics_path = os.path.join(project_root, "models", "arima_metrics.json")
    arima_model_path = os.path.join(project_root, "models", "arima_model.joblib")
    model_exists = os.path.exists(arima_metrics_path) and os.path.exists(
        arima_model_path
    )

    tab_trends, tab_view, tab_train = st.tabs(
        ["📊 Динамика рынка", "🔮 Прогноз SARIMA", "⚙️ Обучение"]
    )

    # =============================================
    # TAB 1: ДИНАМИКА РЫНКА (новая вкладка)
    # =============================================
    with tab_trends:
        st.caption(
            "Временные графики количества вакансий и зарплат. "
            "Выбирайте регион и частоту для детального анализа."
        )

        df, filename = _load_data()

        if df is None:
            st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`.")
        else:
            # Собираем опции из данных
            regions = _get_unique_sorted(df.get("area_name"))
            experiences = _get_unique_sorted(df.get("experience_name"))

            # ---- Локальные фильтры ----
            with st.container(border=True):
                st.subheader("⚙️ Параметры")
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    trend_region = st.multiselect(
                        "Регион",
                        regions if regions else ["Не указано"],
                        default=regions[:3] if len(regions) > 3 else regions,
                        key="trend_region",
                    )
                with fc2:
                    trend_exp = st.multiselect(
                        "Опыт",
                        experiences if experiences else ["Не указано"],
                        default=experiences if experiences else [],
                        key="trend_exp",
                    )
                with fc3:
                    trend_freq = st.selectbox(
                        "Частота",
                        ["W (неделя)", "D (день)", "ME (месяц)"],
                        key="trend_freq",
                    )

            freq_map = {"D (день)": "D", "W (неделя)": "W", "ME (месяц)": "ME"}
            selected_freq = freq_map[trend_freq]

            # ---- График: количество вакансий ----
            st.subheader("📈 Количество вакансий во времени")

            if trend_region and len(regions) > 0:
                # Построить по каждому региону отдельную линию
                all_series = []
                for reg in trend_region:
                    ts_data = _build_vacancy_timeline(
                        df, [reg], trend_exp if trend_exp else None, selected_freq
                    )
                    if ts_data is not None:
                        ts_df, _ = ts_data
                        ts_df["region"] = reg
                        all_series.append(ts_df)

                if all_series:
                    combined = pd.concat(all_series, ignore_index=True)
                    fig = px.line(
                        combined,
                        x="date",
                        y="count",
                        color="region",
                        markers=True,
                        title="",
                    )
                    fig.update_layout(
                        height=450,
                        xaxis_title="Дата",
                        yaxis_title="Количество вакансий",
                        legend_title="Регион",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Таблица с итогами по регионам
                    st.subheader("📋 Сводка по регионам")
                    summary = combined.groupby("region")["count"].agg(
                        ["sum", "mean", "min", "max", "count"]
                    ).round(1)
                    summary.columns = [
                        "Всего вакансий",
                        "Ср. в период",
                        "Мин в период",
                        "Макс в период",
                        "Кол-во периодов",
                    ]
                    st.dataframe(summary, use_container_width=True, height=250)
                else:
                    st.warning("Нет данных по выбранным фильтрам для построения графика.")
            else:
                # Все регионы вместе
                ts_data = _build_vacancy_timeline(
                    df,
                    None,
                    trend_exp if trend_exp else None,
                    selected_freq,
                )
                if ts_data is not None:
                    ts_df, freq_label = ts_data
                    fig = px.line(
                        ts_df,
                        x="date",
                        y="count",
                        markers=True,
                        title="",
                    )
                    fig.update_layout(
                        height=450,
                        xaxis_title="Дата",
                        yaxis_title=f"Количество вакансий ({freq_label.lower()})",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Добавляем скользящее среднее
                    if len(ts_df) >= 3:
                        ts_df["ma_3"] = ts_df["count"].rolling(3, min_periods=1).mean()
                        fig_ma = go.Figure()
                        fig_ma.add_trace(
                            go.Scatter(
                                x=ts_df["date"],
                                y=ts_df["count"],
                                mode="lines+markers",
                                name="Факт",
                                line=dict(color="royalblue", width=2),
                                marker=dict(size=4),
                            )
                        )
                        fig_ma.add_trace(
                            go.Scatter(
                                x=ts_df["date"],
                                y=ts_df["ma_3"],
                                mode="lines",
                                name="Скользящее среднее (3)",
                                line=dict(color="red", width=2, dash="dash"),
                            )
                        )
                        fig_ma.update_layout(
                            height=400,
                            xaxis_title="Дата",
                            yaxis_title=f"Количество вакансий ({freq_label.lower()})",
                            legend=dict(
                                yanchor="top", y=0.99, xanchor="left", x=0.01
                            ),
                        )
                        st.subheader("📉 Сглаженный тренд")
                        st.plotly_chart(fig_ma, use_container_width=True)
                else:
                    st.warning(
                        "Нет данных для построения графика. "
                        "Проверьте, что в данных есть колонка `published_at`."
                    )

            st.divider()

            # ---- График: средняя зарплата во времени ----
            st.subheader("💰 Динамика средней зарплаты")

            sal_data = _build_salary_timeline(
                df,
                trend_region if trend_region else None,
                trend_exp if trend_exp else None,
                selected_freq,
            )

            if sal_data is not None:
                fig_sal = px.line(
                    sal_data,
                    x="date",
                    y="avg_salary",
                    markers=True,
                    title="",
                )
                fig_sal.update_layout(
                    height=400,
                    xaxis_title="Дата",
                    yaxis_title="Средняя зарплата, ₽",
                    showlegend=False,
                )
                st.plotly_chart(fig_sal, use_container_width=True)
            else:
                st.info(
                    "Нет данных о зарплатах для построения графика динамики ЗП. "
                    "Проверьте, что в данных есть колонки `published_at` и `salary_avg`."
                )

    # =============================================
    # TAB 2: ПРОГНОЗ SARIMA
    # =============================================
    with tab_view:
        if not model_exists:
            st.warning(
                "Модель SARIMA ещё не обучена. Перейдите на вкладку "
                "**Обучение** или запустите:\n\n"
                "```\npython scripts/train_models.py --skip-salary --skip-clustering\n```"
            )
        else:
            try:
                with open(arima_metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

                with st.container(border=True):
                    st.subheader("📐 Метрики модели")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AIC", f"{metrics.get('AIC', 0):.2f}")
                    c2.metric("BIC", f"{metrics.get('BIC', 0):.2f}")
                    c3.metric(
                        "Исторический MAE",
                        f"{metrics.get('MAE_historical', 0):.1f}",
                    )

                    order = metrics.get("order", [1, 0, 1])
                    seasonal_order = metrics.get("seasonal_order", [1, 1, 1, 7])
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(
                            f"**Порядок ARIMA (p,d,q):** ({order[0]}, {order[1]}, {order[2]})"
                        )
                    with c2:
                        st.write(
                            f"**Сезонный (P,D,Q,m):** ({seasonal_order[0]}, "
                            f"{seasonal_order[1]}, {seasonal_order[2]}, {seasonal_order[3]})"
                        )
                    with c3:
                        st.write(
                            f"**Точек данных:** {metrics.get('data_points', 'N/A')}"
                        )
                        st.write(
                            f"**Периоды прогноза:** {metrics.get('forecast_periods', 21)}"
                        )
                        if metrics.get("log_transform"):
                            st.write("**Трансформация:** log1p")

                try:
                    from src.ml.arima_analyzer import TimeSeriesAnalyzer

                    analyzer = TimeSeriesAnalyzer()
                    analyzer.load(arima_model_path)

                    df, _ = _load_data()

                    if df is None:
                        st.warning(
                            "Данные не найдены. Поместите CSV в папку `finaldata/`."
                        )
                    else:
                        ts = analyzer.prepare_time_series(
                            df, freq="D", end_date="2026-04-05"
                        )
                        if len(ts) >= 5:
                            forecast_df = analyzer.forecast(
                                periods=metrics.get("forecast_periods", 21)
                            )

                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=ts.index,
                                    y=ts.values,
                                    mode="lines+markers",
                                    name="Исторические данные",
                                    line=dict(color="royalblue", width=2),
                                    marker=dict(size=4),
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_df["date"],
                                    y=forecast_df["forecast"],
                                    mode="lines+markers",
                                    name="Прогноз SARIMA",
                                    line=dict(color="orange", width=2, dash="dash"),
                                    marker=dict(size=4),
                                )
                            )
                            freq_label = (
                                "день"
                                if metrics.get("seasonal_period", 7) == 7
                                else "неделя"
                            )
                            fig.update_layout(
                                title=f"Количество вакансий ({freq_label}): история + прогноз SARIMA",
                                xaxis_title="Дата",
                                yaxis_title="Количество вакансий",
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01,
                                ),
                                height=500,
                            )
                            with st.container(border=True):
                                st.plotly_chart(fig, use_container_width=True)

                            with st.expander("📋 Таблица прогноза", expanded=False):
                                forecast_display = forecast_df.copy()
                                forecast_display["date"] = pd.to_datetime(
                                    forecast_display["date"]
                                ).dt.strftime("%Y-%m-%d")
                                st.dataframe(
                                    forecast_display, use_container_width=True
                                )
                        else:
                            st.warning(
                                "Недостаточно данных для построения графика (менее 5 точек)"
                            )

                except ImportError:
                    st.error(
                        "Библиотека plotly не установлена. Установите: pip install plotly"
                    )
                except Exception as e:
                    st.error(f"Ошибка при построении графика: {e}")
                    import traceback

                    st.code(traceback.format_exc())

                try:
                    from src.ml.arima_analyzer import TimeSeriesAnalyzer

                    analyzer = TimeSeriesAnalyzer()
                    analyzer.load(arima_model_path)
                    stationarity = analyzer._stationarity_result
                    if stationarity:
                        with st.container(border=True):
                            st.subheader("🔬 Анализ стационарности")
                            c1, c2, c3 = st.columns(3)
                            c1.metric(
                                "ADF статистика",
                                f"{stationarity['adf_statistic']:.4f}",
                            )
                            c2.metric(
                                "p-value", f"{stationarity['p_value']:.4f}"
                            )
                            with c3:
                                if stationarity["is_stationary"]:
                                    st.success("Ряд стационарен")
                                else:
                                    st.warning(
                                        "Ряд нестационарен (требуется дифференцирование)"
                                    )
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Ошибка загрузки метрик: {e}")

    # =============================================
    # TAB 3: ОБУЧЕНИЕ SARIMA
    # =============================================
    with tab_train:
        with st.container(border=True):
            st.subheader("Обучение модели SARIMA")

            df, filename = _load_data()

            if df is None:
                st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`.")
                return

            st.success(f"✅ Данные: {filename} ({len(df)} записей)")
            st.caption(
                "Данные обрезаются до 5 апреля 2026 (далее были проблемы со сбором)"
            )

            col1, col2 = st.columns(2)
            with col1:
                freq = st.selectbox(
                    "Частота агрегации",
                    ["D (день)", "W (неделя)", "ME (месяц)"],
                    index=0,
                )
            with col2:
                forecast_periods = st.slider("Периодов прогноза", 1, 60, 21)

            freq_map = {"W (неделя)": "W", "D (день)": "D", "ME (месяц)": "ME"}
            selected_freq = freq_map[freq]

            st.info(
                "**Сезонный период** определяется автоматически: "
                "D→7 (неделя), W→52 (год), M→12 (год)"
            )

            train_btn = st.button(
                "🚀 Обучить SARIMA", type="primary", use_container_width=True
            )

        if train_btn:
            with st.spinner("Обучение SARIMA (с сезонной компонентой)..."):
                try:
                    from src.ml.arima_analyzer import TimeSeriesAnalyzer

                    analyzer = TimeSeriesAnalyzer()
                    ts = analyzer.prepare_time_series(
                        df, freq=selected_freq, end_date="2026-04-05"
                    )

                    st.write(
                        f"Временной ряд: {len(ts)} точек, "
                        f"сезонный период m={analyzer._seasonal_period}"
                    )

                    stationarity = analyzer.check_stationarity(ts)
                    st.write(
                        f"**Стационарность:** "
                        f"{'Да' if stationarity['is_stationary'] else 'Нет'} "
                        f"(p-value: {stationarity['p_value']:.4f})"
                    )

                    result = analyzer.train(ts, forecast_periods=forecast_periods)
                    metrics = result["metrics"]

                    st.success("✅ Модель SARIMA обучена!")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("AIC", f"{metrics['AIC']:.2f}")
                    c2.metric("BIC", f"{metrics['BIC']:.2f}")
                    c3.metric("Исторический MAE", f"{metrics['MAE_historical']:.2f}")

                    order = metrics.get("order", [1, 0, 1])
                    s_order = metrics.get("seasonal_order", [1, 1, 1, 7])
                    st.write(
                        f"**Порядок ARIMA (p,d,q):** ({order[0]}, {order[1]}, {order[2]})"
                    )
                    st.write(
                        f"**Сезонный порядок (P,D,Q,m):** ({s_order[0]}, "
                        f"{s_order[1]}, {s_order[2]}, {s_order[3]})"
                    )
                    if metrics.get("log_transform"):
                        st.write(
                            "**Трансформация:** log1p (для count-данных, прогноз >= 0)"
                        )

                    analyzer.save()
                    st.caption("Модель сохранена в models/arima_model.joblib")

                    st.rerun()

                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
                    import traceback

                    st.code(traceback.format_exc())
