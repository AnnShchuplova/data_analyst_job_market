import json
import os
import pandas as pd
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


def view_arima(service=None):
    st.markdown(
        "<h1 style='text-align: center;'>📈 АНАЛИЗ ТРЕНДОВ (SARIMA)</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Прогнозирование количества вакансий во времени методом SARIMA "
        "(с автоматическим подбором сезонной компоненты)."
    )

    project_root = _project_root()
    arima_metrics_path = os.path.join(project_root, "models", "arima_metrics.json")
    arima_model_path = os.path.join(project_root, "models", "arima_model.joblib")
    model_exists = os.path.exists(arima_metrics_path) and os.path.exists(
        arima_model_path
    )

    tab_view, tab_train = st.tabs(["📊 Результаты SARIMA", "⚙️ Обучение"])

    with tab_view:
        if not model_exists:
            st.warning(
                "⚠️ Модель SARIMA ещё не обучена. Перейдите на вкладку "
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
                    ["D (день)", "W (неделя)", "M (месяц)"],
                    index=0,
                )
            with col2:
                forecast_periods = st.slider("Периодов прогноза", 1, 60, 21)

            freq_map = {"W (неделя)": "W", "D (день)": "D", "M (месяц)": "M"}
            selected_freq = freq_map[freq]

            st.info(
                "💡 **Сезонный период** определяется автоматически: "
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
