import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Прогноз зарплат", layout="wide")

st.title("💰 Предсказатель заработной платы")
st.markdown("---")

st.info("""
> **Прогнозирование заработной платы** 
> ML-модель для предсказания зарплаты аналитика данных
> на основе региона, опыта, навыков и должности.
""")


def load_data():
    """Загрузка данных для обучения модели."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        processed_dir = os.path.join(project_root, "finaldata")
        if os.path.exists(processed_dir):
            files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
                file_path = os.path.join(processed_dir, latest_file)
                df = pd.read_csv(file_path, encoding='utf-8')
                return df, latest_file
        return None, None
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None, None


def main():
    df, filename = load_data()
    
    # Вкладки
    tab_predict, tab_train = st.tabs(["🎯 Прогноз", "⚙️ Обучение модели"])
    
    with tab_train:
        st.subheader("Обучение модели прогноза зарплаты")
        
        if df is None:
            st.warning("Данные не найдены")
            return
        
        st.success(f"✅ Данные: {filename} ({len(df)} записей)")
        
        with_salary = df['has_salary'].sum() if 'has_salary' in df.columns else 0
        st.metric("Записей с указанной ЗП", int(with_salary))
        
        col1, col2 = st.columns(2)
        with col1:
            algorithm = st.selectbox(
                "Алгоритм",
                ["RandomForest", "GradientBoosting", "CatBoost (если установлен)"],
                index=0
            )
        with col2:
            tune_hp = st.checkbox("Подбор гиперпараметров (GridSearch)", value=False)
        
        if st.button("🚀 Обучить модель", type="primary", width="stretch"):
            with st.spinner("Обучение модели..."):
                try:
                    from src.data_processing.feature_engineering import FeatureEngineer
                    from src.ml.salary_predictor import SalaryPredictor
                    
                    fe = FeatureEngineer()
                    df_prepared, feature_cols = fe.prepare_dataset_for_regression(df, fit=True)
                    
                    algo_map = {
                        "RandomForest": "random_forest",
                        "GradientBoosting": "gradient_boosting",
                        "CatBoost (если установлен)": "catboost"
                    }
                    
                    predictor = SalaryPredictor()
                    metrics = predictor.train(
                        df_prepared, feature_cols,
                        algorithm=algo_map[algorithm],
                        tune_hyperparams=tune_hp
                    )
                    
                    st.success("✅ Модель обучена!")
                    
                    # Вывод метрик
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{metrics['MAE']:,.0f} руб.")
                    with col2:
                        st.metric("MAE % от ср. ЗП", f"{metrics['MAE_percent']:.1f}%")
                    with col3:
                        st.metric("R²", f"{metrics['R2']:.4f}")
                    with col4:
                        st.metric("RMSE", f"{metrics['RMSE']:,.0f} руб.")
                    
                    # Проверка соответствия ТЗ
                    if metrics['MAE_percent'] <= 20:
                        st.success("✅ MAE в пределах 20% от средней ЗП (ТЗ выполнено)")
                    else:
                        st.warning(f"⚠️ MAE {metrics['MAE_percent']:.1f}% > 20% (ТЗ не выполнено)")
                    
                    if metrics['R2'] >= 0.7:
                        st.success("✅ R² >= 0.7 (ТЗ выполнено)")
                    else:
                        st.warning(f"⚠️ R² {metrics['R2']:.4f} < 0.7 (ТЗ не выполнено)")
                    
                    # Важность признаков
                    importance = predictor.get_feature_importance()
                    if importance is not None:
                        st.subheader("📊 Важность признаков")
                        fig = px.bar(
                            importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Топ-15 наиболее важных признаков"
                        )
                        st.plotly_chart(fig, width="stretch")
                    
                    # Сохранение
                    try:
                        predictor.save()
                        st.caption("Модель сохранена в models/salary_model.joblib")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить модель: {e}")
                    
                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with tab_predict:
        st.subheader("📝 Параметры вакансии")
        
        with st.form("salary_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                experience = st.selectbox(
                    "Опыт работы",
                    ["Junior (до 1 года)", "Middle (1-3 года)", "Senior (3+ лет)"],
                    index=1
                )
                
                region = st.selectbox(
                    "Регион",
                    [
                        "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург",
                        "Казань", "Нижний Новгород", "Краснодар", "Самара",
                        "Воронеж", "Пермь", "Другой"
                    ],
                    index=0
                )
            
            with col2:
                skills = st.multiselect(
                    "Ключевые навыки",
                    [
                        "Python", "SQL", "Excel", "Tableau", "Power BI",
                        "Statistics", "Machine Learning", "A/B testing",
                        "Data Visualization", "ETL", "R", "SAS", "SPSS"
                    ],
                    default=["Python", "SQL"]
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
                    index=0
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
                    index=0
                )
            
            submitted = st.form_submit_button("🎯 Предсказать зарплату")
            
            if submitted:
                with st.spinner("Выполняю прогноз..."):
                    try:
                        from src.ml.salary_predictor import SalaryPredictor
                        
                        models_dir = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "models"
                        )
                        model_path = os.path.join(models_dir, "salary_model.joblib")
                        
                        if not os.path.exists(model_path):
                            st.warning("⚠️ Модель не обучена. Перейдите на вкладку **Обучение модели** для обучения.")
                        else:
                            predictor = SalaryPredictor()
                            predictor.load(model_path)
                            
                            # Подготовка входных данных
                            exp_map = {
                                "Junior (до 1 года)": "Junior",
                                "Middle (1-3 года)": "Middle",
                                "Senior (3+ лет)": "Senior"
                            }
                            
                            experience_level = exp_map[experience]
                            
                            # Опыт в годах
                            exp_years_map = {"Junior": 0, "Middle": 2, "Senior": 5}
                            
                            # CatBoost работает с сырыми строковыми категориями
                            exp_level = exp_map[experience]
                            
                            # Регион: если "Другой" — ставим значение по умолчанию
                            area_val = region if region != 'Другой' else 'Не указан'
                            
                            feature_dict = {
                                # Категориальные признаки (строки для CatBoost)
                                'experience_level': exp_level,
                                'area_name': area_val,
                                'main_role_name': position if position else 'Аналитик',
                                'schedule_name': schedule,
                                # Числовые признаки
                                'skills_count': len(skills),
                                'role_mean_salary': 150000,
                                'region_mean_salary': 150000,
                                'employment_mean_salary': 150000,
                                'work_format_mean_salary': 150000,
                                'has_sql': int('SQL' in skills),
                                'has_python': int('Python' in skills),
                                'has_bi': int(any(s in skills for s in ['Tableau', 'Power BI'])),
                                'has_ml': int('Machine Learning' in skills),
                                'has_stats': int('Statistics' in skills or 'A/B testing' in skills),
                                'has_etl': int('ETL' in skills),
                                'has_cloud': 0,
                                'has_excel': int('Excel' in skills),
                                'key_skills_total': len(skills),
                                'name_has_senior': int(exp_level == 'Senior'),
                                'name_has_junior': int(exp_level == 'Junior'),
                                'is_remote': 0,
                                'is_senior': int(exp_level == 'Senior'),
                                'text_length': 500,
                            }
                            
                            # Заполняем TF-IDF признаки нулями (для одиночного прогноза)
                            if predictor.feature_columns:
                                for col in predictor.feature_columns:
                                    if col.startswith('skill_tfidf_'):
                                        feature_dict[col] = 0.0
                            
                            prediction = predictor.predict_single(feature_dict)
                            
                            # Доверительный интервал (оценка)
                            mae = predictor.metrics.get('MAE', 15000)
                            lower = max(0, int(prediction - mae))
                            upper = int(prediction + mae)
                            
                            # Форматирование
                            pred_fmt = f"{int(prediction):,} ₽".replace(",", " ")
                            range_fmt = f"{lower:,} - {upper:,} ₽".replace(",", " ")
                            
                            st.success("### Прогноз заработной платы")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Прогноз", pred_fmt)
                            with col2:
                                st.metric("Диапазон", range_fmt)
                            with col3:
                                st.metric("Модель", predictor.metrics.get('algorithm', 'N/A'))
                            
                            st.markdown("---")
                            st.subheader("📊 Метрики модели")
                            for k, v in predictor.metrics.items():
                                if k not in ('timestamp', 'algorithm'):
                                    st.write(f"- **{k}**: {v}")
                    
                    except Exception as e:
                        st.error(f"Ошибка прогноза: {e}")
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
