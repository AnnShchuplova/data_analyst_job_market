import streamlit as st

def view_salary_predictor(service):
    st.markdown("<h1 style='text-align: center;'>ПРЕДСКАЗАТЕЛЬ ЗАРПЛАТ</h1>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 2], gap="large")

    # Левая колонка: Ввод данных
    with col_input:
        with st.container(border=True):
            st.subheader("⚙️ Параметры")

            job_title = st.text_input("Название вакансии", value="Data Analyst")

            exp = st.slider("Опыт работы (лет)", 0, 10, 3)

            location = st.selectbox("Местоположение", ["Москва", "Санкт-Петербург", "Удаленно", "Релокация"])

            st.write("Навыки")
            # st.pills доступен в Streamlit 1.40+, если версия старая - используйте multiselect
            skills = st.multiselect("Выберите стек",
                                    ["Python", "SQL", "Tableau", "PowerBI", "Spark", "Docker"],
                                    default=["Python", "SQL"])

            st.write("")
            calc_btn = st.button("РАССЧИТАТЬ ЗАРПЛАТУ", type="primary", use_container_width=True)

    # Правая колонка: Результаты
    with col_result:
        with st.container(border=True):
            if calc_btn:
                with st.spinner("Анализируем рынок..."):
                    result = service.predict_salary(job_title, exp, skills)

                st.subheader("Результат оценки")

                # Основная цифра
                st.markdown(f"<h1 style='font-size: 60px; margin:0;'>{result.currency}{result.predicted_salary:,}</h1>",
                            unsafe_allow_html=True)
                st.caption(
                    f"Доверительный интервал: {result.currency}{result.confidence_interval[0]:,} - {result.currency}{result.confidence_interval[1]:,}")

                st.divider()

                st.write("**Позиция относительно рынка:**")
                # График распределения (Histogram)
                st.bar_chart(result.market_comparison_chart, height=250)

                st.success(f"Вы можете претендовать на зарплату выше средней для {exp} лет опыта!")
            else:
                # Заглушка до нажатия кнопки (пустое состояние)
                st.info("👈 Заполните параметры слева и нажмите кнопку расчета, чтобы увидеть график.")
                for _ in range(8): st.write("")  # Просто заполнить место