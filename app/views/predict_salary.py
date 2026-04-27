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


def _project_root():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def view_salary_predictor(service=None):
    st.markdown(
        "<h1 style='text-align: center;'>💰 ПРЕДСКАЗАТЕЛЬ ЗАРПЛАТ</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "ML-модель для предсказания зарплаты аналитика по региону, опыту, навыкам и должности."
    )

    try:
        df, filename = _load_data()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return

    tab_predict, tab_train = st.tabs(["🎯 Прогноз", "⚙️ Обучение модели"])

    with tab_predict:
        with st.container(border=True):
            st.subheader("📝 Параметры вакансии")
            with st.form("salary_prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    experience = st.selectbox(
                        "Опыт работы",
                        [
                            "Junior (до 1 года)",
                            "Middle (1-3 года)",
                            "Senior (3+ лет)",
                        ],
                        index=1,
                    )
                    region = st.selectbox(
                        "Регион",
                        [
                            "Москва",
                            "Санкт-Петербург",
                            "Новосибирск",
                            "Екатеринбург",
                            "Казань",
                            "Нижний Новгород",
                            "Краснодар",
                            "Самара",
                            "Воронеж",
                            "Пермь",
                            "Другой",
                        ],
                        index=0,
                    )
                with col2:
                    skills = st.multiselect(
                        "Ключевые навыки",
                        [
                            "Python",
                            "SQL",
                            "Excel",
                            "Tableau",
                            "Power BI",
                            "Statistics",
                            "Machine Learning",
                            "A/B testing",
                            "Data Visualization",
                            "ETL",
                            "R",
                            "SAS",
                            "SPSS",
                        ],
                        default=["Python", "SQL"],
                    )
                    position = st.selectbox(
                        "Профессия",
                        [
                            "Аналитик",
                            "BI-аналитик, аналитик данных",
                            "Бизнес-аналитик",
                            "Маркетолог-аналитик",
                            "Продуктовый аналитик",
                            "Системный аналитик",
                            "Финансовый аналитик, инвестиционный аналитик",
                        ],
                        index=0,
                    )
                    schedule = st.selectbox(
                        "Формат работы",
                        [
                            "Полный день",
                            "Удаленная работа",
                            "Гибкий график",
                            "Сменный график",
                            "Вахтовый метод",
                        ],
                        index=0,
                    )

                submitted = st.form_submit_button(
                    "🎯 Предсказать зарплату", type="primary", use_container_width=True
                )

        if submitted:
            with st.spinner("Выполняю прогноз..."):
                try:
                    from src.ml.salary_predictor import SalaryPredictor

                    model_path = os.path.join(
                        _project_root(), "models", "salary_model.joblib"
                    )

                    if not os.path.exists(model_path):
                        st.warning(
                            "⚠️ Модель не обучена. Перейдите на вкладку **Обучение модели**."
                        )
                    else:
                        predictor = SalaryPredictor()
                        predictor.load(model_path)

                        exp_map = {
                            "Junior (до 1 года)": "Junior",
                            "Middle (1-3 года)": "Middle",
                            "Senior (3+ лет)": "Senior",
                        }
                        exp_level = exp_map[experience]
                        area_val = region if region != "Другой" else "Не указан"

                        feature_dict = {
                            "experience_level": exp_level,
                            "area_name": area_val,
                            "main_role_name": position if position else "Аналитик",
                            "schedule_name": schedule,
                            "skills_count": len(skills),
                            "role_mean_salary": 150000,
                            "region_mean_salary": 150000,
                            "employment_mean_salary": 150000,
                            "work_format_mean_salary": 150000,
                            "has_sql": int("SQL" in skills),
                            "has_python": int("Python" in skills),
                            "has_bi": int(
                                any(s in skills for s in ["Tableau", "Power BI"])
                            ),
                            "has_ml": int("Machine Learning" in skills),
                            "has_stats": int(
                                "Statistics" in skills or "A/B testing" in skills
                            ),
                            "has_etl": int("ETL" in skills),
                            "has_cloud": 0,
                            "has_excel": int("Excel" in skills),
                            "key_skills_total": len(skills),
                            "name_has_senior": int(exp_level == "Senior"),
                            "name_has_junior": int(exp_level == "Junior"),
                            "is_remote": 0,
                            "is_senior": int(exp_level == "Senior"),
                            "text_length": 500,
                        }

                        if predictor.feature_columns:
                            for col in predictor.feature_columns:
                                if col.startswith("skill_tfidf_"):
                                    feature_dict[col] = 0.0

                        prediction = predictor.predict_single(feature_dict)

                        mae = predictor.metrics.get("MAE", 15000)
                        lower = max(0, int(prediction - mae))
                        upper = int(prediction + mae)

                        pred_fmt = f"{int(prediction):,} ₽".replace(",", " ")
                        range_fmt = f"{lower:,} - {upper:,} ₽".replace(",", " ")

                        with st.container(border=True):
                            st.subheader("Результат оценки")
                            st.markdown(
                                f"<h1 style='font-size: 60px; margin:0;'>{pred_fmt}</h1>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Доверительный интервал: {range_fmt}")
                            st.divider()
                            c1, c2 = st.columns(2)
                            c1.metric(
                                "Модель",
                                predictor.metrics.get("algorithm", "N/A"),
                            )
                            c2.metric("MAE модели", f"{int(mae):,} ₽".replace(",", " "))

                            st.markdown("**📊 Метрики модели:**")
                            for k, v in predictor.metrics.items():
                                if k not in ("timestamp", "algorithm"):
                                    st.write(f"- **{k}**: {v}")
                except Exception as e:
                    st.error(f"Ошибка прогноза: {e}")
                    import traceback

                    st.code(traceback.format_exc())
        else:
            st.info("👆 Заполните параметры выше и нажмите кнопку расчета.")

    with tab_train:
        with st.container(border=True):
            st.subheader("Обучение модели прогноза зарплаты")

            if df is None:
                st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`.")
                return

            st.success(f"✅ Данные: {filename} ({len(df)} записей)")

            with_salary = (
                df["has_salary"].sum() if "has_salary" in df.columns else 0
            )
            st.metric("Записей с указанной ЗП", int(with_salary))

            col1, col2 = st.columns(2)
            with col1:
                algorithm = st.selectbox(
                    "Алгоритм",
                    [
                        "RandomForest",
                        "GradientBoosting",
                        "CatBoost (если установлен)",
                    ],
                    index=0,
                )
            with col2:
                tune_hp = st.checkbox(
                    "Подбор гиперпараметров (GridSearch)", value=False
                )

            train_btn = st.button(
                "🚀 Обучить модель", type="primary", use_container_width=True
            )

        if train_btn:
            with st.spinner("Обучение модели..."):
                try:
                    from src.data_processing.feature_engineering import (
                        FeatureEngineer,
                    )
                    from src.ml.salary_predictor import SalaryPredictor

                    fe = FeatureEngineer()
                    df_prepared, feature_cols = fe.prepare_dataset_for_regression(
                        df, fit=True
                    )

                    algo_map = {
                        "RandomForest": "random_forest",
                        "GradientBoosting": "gradient_boosting",
                        "CatBoost (если установлен)": "catboost",
                    }

                    predictor = SalaryPredictor()
                    metrics = predictor.train(
                        df_prepared,
                        feature_cols,
                        algorithm=algo_map[algorithm],
                        tune_hyperparams=tune_hp,
                    )

                    st.success("✅ Модель обучена!")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MAE", f"{metrics['MAE']:,.0f} руб.")
                    c2.metric("MAE % от ср. ЗП", f"{metrics['MAE_percent']:.1f}%")
                    c3.metric("R²", f"{metrics['R2']:.4f}")
                    c4.metric("RMSE", f"{metrics['RMSE']:,.0f} руб.")

                    if metrics["MAE_percent"] <= 20:
                        st.success("✅ MAE в пределах 20% от средней ЗП (ТЗ выполнено)")
                    else:
                        st.warning(
                            f"⚠️ MAE {metrics['MAE_percent']:.1f}% > 20% (ТЗ не выполнено)"
                        )

                    if metrics["R2"] >= 0.7:
                        st.success("✅ R² >= 0.7 (ТЗ выполнено)")
                    else:
                        st.warning(
                            f"⚠️ R² {metrics['R2']:.4f} < 0.7 (ТЗ не выполнено)"
                        )

                    importance = predictor.get_feature_importance()
                    if importance is not None:
                        with st.container(border=True):
                            st.subheader("📊 Важность признаков")
                            fig = px.bar(
                                importance,
                                x="importance",
                                y="feature",
                                orientation="h",
                                title="Топ-15 наиболее важных признаков",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    try:
                        predictor.save()
                        st.caption("Модель сохранена в models/salary_model.joblib")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить модель: {e}")

                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
                    import traceback

                    st.code(traceback.format_exc())
