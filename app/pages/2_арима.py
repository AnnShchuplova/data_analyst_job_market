import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Анализ трендов (SARIMA)", layout="wide")

st.title("📈 Анализ трендов вакансий (SARIMA)")
st.markdown("---")

st.info("""
> **Прогнозирование временных рядов**
""")


def load_data():
    """Загрузка данных из finaldata/."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "finaldata")
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
                file_path = os.path.join(data_dir, latest_file)
                df = pd.read_csv(file_path, encoding='utf-8')
                return df, latest_file
        return None, None
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None, None


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arima_metrics_path = os.path.join(project_root, "models", "arima_metrics.json")
    arima_model_path = os.path.join(project_root, "models", "arima_model.joblib")

    model_exists = os.path.exists(arima_metrics_path) and os.path.exists(arima_model_path)

    tab_view, tab_train = st.tabs(["📊 Результаты SARIMA", "⚙️ Обучение"])

    # ===== Вкладка "Результаты SARIMA" =====
    with tab_view:
        if not model_exists:
            st.warning("⚠️ Модель SARIMA ещё не обучена. Перейдите на вкладку **Обучение** или запустите:\n"
                        "```\npython scripts/train_models.py --skip-salary --skip-clustering\n```")
        else:
            try:
                import json
                with open(arima_metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{metrics.get('AIC', 0):.2f}")
                with col2:
                    st.metric("BIC", f"{metrics.get('BIC', 0):.2f}")
                with col3:
                    st.metric("Исторический MAE", f"{metrics.get('MAE_historical', 0):.1f}")

                st.markdown("---")

                col1, col2, col3 = st.columns(3)
                with col1:
                    order = metrics.get('order', [1, 0, 1])
                    st.write(f"**Порядок ARIMA (p,d,q):** ({order[0]}, {order[1]}, {order[2]})")
                with col2:
                    seasonal_order = metrics.get('seasonal_order', [1, 1, 1, 7])
                    st.write(f"**Сезонный (P,D,Q,m):** ({seasonal_order[0]}, {seasonal_order[1]}, {seasonal_order[2]}, {seasonal_order[3]})")
                with col3:
                    st.write(f"**Точек данных:** {metrics.get('data_points', 'N/A')}")
                    st.write(f"**Периоды прогноза:** {metrics.get('forecast_periods', 21)}")
                    if metrics.get('log_transform'):
                        st.write("**Трансформация:** log1p")

                # Строим график
                try:
                    from src.ml.arima_analyzer import TimeSeriesAnalyzer
                    import plotly.graph_objects as go

                    analyzer = TimeSeriesAnalyzer()
                    analyzer.load(arima_model_path)

                    df, filename = load_data()

                    if df is not None:
                        ts = analyzer.prepare_time_series(df, freq='D', end_date='2026-04-05')

                        if len(ts) >= 5:
                            forecast_df = analyzer.forecast(periods=metrics.get('forecast_periods', 21))

                            fig = go.Figure()

                            # Исторические данные
                            fig.add_trace(go.Scatter(
                                x=ts.index,
                                y=ts.values,
                                mode='lines+markers',
                                name='Исторические данные',
                                line=dict(color='royalblue', width=2),
                                marker=dict(size=4)
                            ))

                            # Прогноз
                            fig.add_trace(go.Scatter(
                                x=forecast_df['date'],
                                y=forecast_df['forecast'],
                                mode='lines+markers',
                                name='Прогноз SARIMA',
                                line=dict(color='orange', width=2, dash='dash'),
                                marker=dict(size=4)
                            ))

                

                            freq_label = 'день' if metrics.get('seasonal_period', 7) == 7 else 'неделя'
                            fig.update_layout(
                                title=f'Количество вакансий ({freq_label}): исторические + прогноз SARIMA',
                                xaxis_title='Дата',
                                yaxis_title='Количество вакансий',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                height=500
                            )

                            st.plotly_chart(fig)

                            # Таблица прогноза
                            st.subheader("📋 Таблица прогноза")
                            forecast_display = forecast_df.copy()
                            forecast_display['date'] = pd.to_datetime(forecast_display['date']).dt.strftime('%Y-%m-%d')
                            st.dataframe(forecast_display)
                        else:
                            st.warning("Недостаточно данных для построения графика (менее 5 точек)")
                    else:
                        st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`")

                except ImportError:
                    st.error("Библиотека plotly не установлена. Установите: pip install plotly")
                except Exception as e:
                    st.error(f"Ошибка при построении графика: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Ошибка загрузки метрик: {e}")

        # Стационарность
        if model_exists:
            st.markdown("---")
            st.subheader("🔬 Анализ стационарности")
            try:
                from src.ml.arima_analyzer import TimeSeriesAnalyzer
                analyzer = TimeSeriesAnalyzer()
                analyzer.load(arima_model_path)
                stationarity = analyzer._stationarity_result
                if stationarity:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ADF статистика", f"{stationarity['adf_statistic']:.4f}")
                    with col2:
                        st.metric("p-value", f"{stationarity['p_value']:.4f}")
                    with col3:
                        if stationarity['is_stationary']:
                            st.success("Ряд стационарен")
                        else:
                            st.warning("Ряд нестационарен (требуется дифференцирование)")
            except Exception:
                st.write("Информация о стационарности недоступна")

    # ===== Вкладка "Обучение" =====
    with tab_train:
        st.subheader("Обучение модели SARIMA")

        df, filename = load_data()

        if df is None:
            st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`")
        else:
            st.success(f"✅ Данные: {filename} ({len(df)} записей)")
            st.caption("Данные обрезаются до 5 апреля 2026 (далее были проблемы со сбором)")

            col1, col2 = st.columns(2)
            with col1:
                freq = st.selectbox("Частота агрегации", ["D (день)", "W (неделя)", "M (месяц)"], index=0)
            with col2:
                forecast_periods = st.slider("Периодов прогноза", 1, 60, 21)

            freq_map = {"W (неделя)": "W", "D (день)": "D", "M (месяц)": "M"}
            selected_freq = freq_map[freq]

            st.info(f"💡 **Сезонный период** будет определён автоматически: D→7 (недельная цикличность), W→52 (годичная), M→12 (годичная)")

            if st.button("🚀 Обучить SARIMA", type="primary"):
                with st.spinner("Обучение SARIMA (с сезонной компонентой)..."):
                    try:
                        from src.ml.arima_analyzer import TimeSeriesAnalyzer

                        analyzer = TimeSeriesAnalyzer()
                        ts = analyzer.prepare_time_series(df, freq=selected_freq, end_date='2026-04-05')

                        st.write(f"Временной ряд: {len(ts)} точек, сезонный период m={analyzer._seasonal_period}")

                        # Стационарность
                        stationarity = analyzer.check_stationarity(ts)
                        st.write(f"**Стационарность:** {'Да' if stationarity['is_stationary'] else 'Нет'} "
                                f"(p-value: {stationarity['p_value']:.4f})")

                        # Обучение
                        result = analyzer.train(ts, forecast_periods=forecast_periods)

                        metrics = result['metrics']
                        st.success("✅ Модель SARIMA обучена!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("AIC", f"{metrics['AIC']:.2f}")
                        with col2:
                            st.metric("BIC", f"{metrics['BIC']:.2f}")
                        with col3:
                            st.metric("Исторический MAE", f"{metrics['MAE_historical']:.2f}")

                        order = metrics.get('order', [1, 0, 1])
                        s_order = metrics.get('seasonal_order', [1, 1, 1, 7])
                        st.write(f"**Порядок ARIMA (p,d,q):** ({order[0]}, {order[1]}, {order[2]})")
                        st.write(f"**Сезонный порядок (P,D,Q,m):** ({s_order[0]}, {s_order[1]}, {s_order[2]}, {s_order[3]})")
                        if metrics.get('log_transform'):
                            st.write("**Трансформация:** log1p (для count-данных, прогноз всегда >= 0)")

                        # Сохранение
                        analyzer.save()
                        st.caption("Модель сохранена в models/arima_model.joblib")

                        # Перерисовать
                        st.rerun()

                    except Exception as e:
                        st.error(f"Ошибка обучения: {e}")
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
