import os
import traceback
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st


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


def _project_root():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


@st.cache_resource
def _load_model():
    """Загрузка модели с кешированием — загружается ОДИН РАЗ."""
    model_path = os.path.join(_project_root(), "models", "salary_predictor.pkl")
    if not os.path.exists(model_path):
        return None
    from src.ml.salary_predictor import SalaryPredictor
    return SalaryPredictor.load(model_path)


def _get_unique_sorted(series):
    """Уникальные непустые значения, отсортированные по частоте."""
    if series is None:
        return []
    counts = series.dropna().value_counts()
    return list(counts.index)


def _get_top_skills(df, n=25):
    """Извлечение топ-навыков из skills_list, отсортированных по частоте."""
    all_skills = []
    if df is not None and "skills_list" in df.columns:
        for raw in df["skills_list"].dropna():
            if isinstance(raw, str):
                text = raw.strip()
                if text.startswith("[") and text.endswith("]"):
                    text = text[1:-1]
                for part in text.split(","):
                    s = part.strip().strip("'\"").strip()
                    if s and len(s) > 1:
                        all_skills.append(s)
    counts = Counter(all_skills)
    return [skill for skill, _ in counts.most_common(n)]


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

    # Загружаем модель заранее (кешируется)
    predictor = _load_model()

    tab_predict, tab_train = st.tabs(["🎯 Прогноз", "⚙️ Обучение модели"])

    # =============================================
    # Вкладка ПРОГНОЗ
    # =============================================
    with tab_predict:
        # --- Опции из данных ---
        regions = []
        roles = []
        experiences = []
        schedules = []
        work_formats = []
        employments = []

        if df is not None:
            regions = _get_unique_sorted(df.get("area_name"))
            roles = _get_unique_sorted(df.get("main_role_name"))
            experiences = _get_unique_sorted(df.get("experience_name"))
            schedules = _get_unique_sorted(df.get("schedule_name"))
            work_formats = _get_unique_sorted(df.get("work_format_name"))
            employments = _get_unique_sorted(df.get("employment_name"))

        top_skills = _get_top_skills(df, n=25)

        with st.container(border=True):
            st.subheader("📝 Параметры вакансии")
            with st.form("salary_prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    experience = st.selectbox(
                        "Опыт работы",
                        experiences if experiences else ["Не указано"],
                    )
                    region = st.selectbox(
                        "Регион",
                        regions if regions else ["Москва"],
                    )
                    employment = st.selectbox(
                        "Тип занятости",
                        employments if employments else ["Полная занятость"],
                    )
                with col2:
                    skills = st.multiselect(
                        "Ключевые навыки",
                        top_skills if top_skills else ["python", "sql"],
                    )
                    position = st.selectbox(
                        "Профессия",
                        roles if roles else ["Аналитик"],
                    )

                col3, col4 = st.columns(2)
                with col3:
                    schedule = st.selectbox(
                        "График работы",
                        schedules if schedules else ["Полный день"],
                    )
                with col4:
                    work_format = st.selectbox(
                        "Формат работы",
                        work_formats if work_formats else ["Удаленная работа"],
                    )

                submitted = st.form_submit_button(
                    "🎯 Предсказать зарплату", type="primary", use_container_width=True
                )

        if submitted:
            if predictor is None:
                st.warning(
                    "⚠️ Модель не найдена. Перейдите на вкладку "
                    "**Обучение модели** и обучите её."
                )
            else:
                with st.spinner("Выполняю прогноз..."):
                    try:
                        row = {
                            "experience_name": experience,
                            "area_name": region,
                            "main_role_name": position,
                            "schedule_name": schedule,
                            "work_format_name": work_format,
                            "employment_name": employment,
                            "skills_list": ", ".join(skills) if skills else "",
                            "name": f"{position} {experience}",
                            "has_test": 0,
                            "premium": 0,
                            "response_letter_required": 0,
                            "accept_labor_contract": 1,
                            "experience_id": 1,
                            "min_experience_years": 1.0,
                            "avg_experience_years": 2.0,
                            "estimated_experience_years": 2.0,
                            "skills_count": len(skills),
                            "days_since_publication": 0,
                            "requirement": "",
                            "responsibility": "",
                            "employer_name": "",
                        }

                        input_df = pd.DataFrame([row])

                        if predictor.feature_engineer is not None:
                            features_df, _ = predictor.feature_engineer.transform(input_df)
                        else:
                            st.error("Модель не содержит feature_engineer. Переобучите.")
                            return

                        prediction = predictor.predict(features_df)
                        salary = float(prediction[0]) if prediction else 0

                        best = predictor.best_model_name or "CatBoost"
                        mae = predictor.best_mae if predictor.best_mae else 20000

                        st.markdown("---")
                        with st.container(border=True):
                            st.subheader("Результат оценки")
                            pred_fmt = f"{int(salary):,} ₽".replace(",", " ")
                            st.markdown(
                                f"<h1 style='font-size: 60px; margin:0;'>{pred_fmt}</h1>",
                                unsafe_allow_html=True,
                            )
                            lower = max(0, int(salary - mae))
                            upper = int(salary + mae)
                            range_fmt = f"{lower:,} - {upper:,} ₽".replace(",", " ")
                            st.caption(f"Доверительный интервал (±MAE): {range_fmt}")

                            st.divider()
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Лучшая модель", best)
                            c2.metric("MAE", f"{int(mae):,} ₽".replace(",", " "))
                            if predictor.best_r2 is not None:
                                c3.metric("R²", f"{predictor.best_r2:.4f}")

                    except Exception as e:
                        st.error(f"Ошибка прогноза: {e}")
                        st.code(traceback.format_exc())
        else:
            if df is not None:
                n_with_salary = int(df["has_salary"].sum()) if "has_salary" in df.columns else "?"
                st.info(
                    f"👆 Заполните параметры и нажмите кнопку. "
                    f"В базе {len(df)} вакансий, с указанной ЗП: {n_with_salary}."
                )

    # =============================================
    # Вкладка ОБУЧЕНИЕ
    # =============================================
    with tab_train:
        with st.container(border=True):
            st.subheader("Обучение модели прогноза зарплаты")

            if df is None:
                st.warning("Данные не найдены. Поместите CSV в папку `finaldata/`.")
                return

            st.success(f"✅ Данные: {filename} ({len(df)} записей)")

            has_salary_col = "has_salary" in df.columns
            salary_col = "salary_avg" in df.columns

            if not salary_col:
                st.error("В данных нет колонки `salary_avg`.")
                return

            with_salary = int(df["has_salary"].sum()) if has_salary_col else len(df)
            st.metric("Записей с указанной ЗП", with_salary)

            filtered = df[df["salary_avg"] > 0].copy() if salary_col else df.copy()
            if has_salary_col:
                filtered = filtered[filtered["has_salary"] == True]

            abs_min = 15000
            filtered = filtered[filtered["salary_avg"] >= abs_min]

            Q1 = filtered["salary_avg"].quantile(0.25)
            Q3 = filtered["salary_avg"].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            filtered = filtered[filtered["salary_avg"] <= upper_bound]

            st.info(
                f"После фильтрации: {len(filtered)} записей "
                f"(ЗП от {int(filtered['salary_avg'].min())} до {int(filtered['salary_avg'].max())}₽)"
            )

            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Размер тестовой выборки", 0.1, 0.3, 0.2, 0.05)
            with col2:
                st.markdown("**Модели:**")
                st.write("- RandomForest")
                st.write("- CatBoost (с text_features)")
                st.write("- GradientBoosting")
                st.write("- Ансамбль (R²-взвешенный)")

            train_btn = st.button(
                "🚀 Обучить модель", type="primary", use_container_width=True
            )

        if train_btn:
            with st.spinner("Обучение моделей (это может занять несколько минут)..."):
                try:
                    from src.data_processing.feature_engineering import FeatureEngineer
                    from src.ml.salary_predictor import SalaryPredictor

                    fe = FeatureEngineer()
                    features_df, top_skill_cols = fe.fit_transform(filtered)

                    y = filtered["salary_avg"]
                    X = features_df.drop(columns=["salary_avg"], errors="ignore")

                    st.info(f"Признаков: {len(X.columns)}, Строк: {len(X)}")

                    predictor = SalaryPredictor()
                    predictor.feature_engineer = fe
                    all_metrics = predictor.compare(
                        X, y,
                        top_skill_cols=top_skill_cols,
                        test_size=test_size,
                    )

                    st.success("✅ Обучение завершено!")

                    best_name = predictor.best_model_name
                    best_r2 = all_metrics[best_name]["R2"]
                    best_mae = all_metrics[best_name]["MAE"]
                    best_mae_pct = all_metrics[best_name]["MAE%"]

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Лучшая модель", best_name)
                    c2.metric("MAE", f"{int(best_mae):,} ₽".replace(",", " "))
                    c3.metric("MAE %", f"{best_mae_pct:.1f}%")
                    c4.metric("R²", f"{best_r2:.4f}")

                    if best_mae_pct <= 20:
                        st.success("✅ MAE ≤ 20% от средней ЗП — ТЗ ВЫПОЛНЕНО")
                    else:
                        st.warning(f"⚠️ MAE {best_mae_pct:.1f}% > 20% — ТЗ не выполнено")

                    if best_r2 >= 0.7:
                        st.success("✅ R² ≥ 0.7 — ТЗ ВЫПОЛНЕНО")
                    else:
                        st.warning(f"⚠️ R² {best_r2:.4f} < 0.7 — ТЗ не выполнено")

                    st.markdown("---")
                    st.subheader("📊 Сравнение моделей")
                    metrics_table = pd.DataFrame(all_metrics).T
                    metrics_table.index.name = "Модель"
                    st.dataframe(metrics_table[["R2", "MAE", "MAE%", "RMSE"]], use_container_width=True)

                    cb_model = predictor.models.get("catboost")
                    if cb_model is not None and hasattr(cb_model, "feature_importances_"):
                        with st.container(border=True):
                            st.subheader("📊 Важность признаков (CatBoost)")
                            cols = predictor.models.get("_cat_columns", list(X.columns))
                            importance_data = pd.DataFrame({
                                "feature": cols,
                                "importance": cb_model.feature_importances_
                            }).sort_values("importance", ascending=False).head(20)

                            fig = px.bar(
                                importance_data,
                                x="importance",
                                y="feature",
                                orientation="h",
                                title="Топ-20 признаков",
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)

                    model_path = os.path.join(_project_root(), "models", "salary_predictor.pkl")
                    try:
                        predictor.save(model_path)
                        _load_model.clear()
                        st.success(f"✅ Модель сохранена: `models/salary_predictor.pkl`")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить модель: {e}")

                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
                    st.code(traceback.format_exc())