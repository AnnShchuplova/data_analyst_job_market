# PROJECT_CONTEXT — DataTrack

**Система:** DataTrack — аналитический веб-сервис рынка труда аналитиков данных  
**Команда:** НИУ ВШЭ, ФКН, БПИ247 — Тищенко Н.А. (2026)  
**Источник данных:** hh.ru API  

---

## 1. Назначение

DataTrack собирает вакансии для аналитиков данных с hh.ru, очищает их, строит ML-модели
и предоставляет интерактивный веб-дашборд для исследования рынка труда:
агрегация → кластеризация → временные тренды (SARIMA) → прогноз зарплаты (CatBoost-ансамбль).

---

## 2. Стек технологий

| Категория | Библиотека / Инструмент |
|-----------|------------------------|
| UI | Streamlit |
| ML / числовые данные | scikit-learn, CatBoost, statsmodels |
| Кластеризация | scikit-learn (K-Means), kmodes (K-Modes, K-Prototypes), gower |
| Визуализация | Plotly, matplotlib, seaborn |
| Данные | pandas, numpy |
| HTTP-клиент | requests (с retry / exponential backoff) |
| Сериализация | pickle (SalaryPredictor), joblib (ARIMA, ClusteringCache) |
| Деплой | Docker Compose (2 сервиса) |
| Тесты | pytest, unittest |

---

## 3. Структура директорий

```
data_analyst_job_market/
├── app/
│   ├── main.py                      # Точка входа Streamlit, роутер страниц
│   └── views/
│       ├── home.py                  # Главная: сводка по БД
│       ├── aggregation.py           # Агрегация (6 вкладок + сайдбар)
│       ├── clusters.py              # Кластеризация вакансий
│       ├── arima.py                 # Тренды + SARIMA прогноз
│       └── predict_salary.py        # Прогноз зарплаты (обучение + предсказание)
├── src/
│   ├── domain/models.py             # Датаклассы: HomeStats, ClusterEntity, ...
│   ├── infrastructure/
│   │   ├── vacancies_repository.py  # CSV-доступ + MD5 checksum
│   │   └── cache.py                 # Кэш кластеризации (joblib)
│   ├── services/
│   │   ├── home_service.py          # Вычисление статистики главной страницы
│   │   ├── clustering_service.py    # Оркестрация K-Means/Modes/Prototypes
│   │   └── mock.py                  # Заглушка (legacy, не используется)
│   ├── data_collection/
│   │   ├── hh_parser.py             # hh.ru API-клиент
│   │   └── filters.py               # Фильтр аналитических ролей
│   ├── data_processing/
│   │   ├── cleaner.py               # JSON → DataFrame, нормализация полей
│   │   └── feature_engineering.py   # Фичи для ML (TF-IDF, target encoding, ...)
│   └── ml/
│       ├── salary_predictor.py      # Ансамбль: RF + CatBoost + GB
│       └── arima_analyzer.py        # SARIMA временной ряд
├── scripts/
│   ├── run_pipeline.py              # Сбор → фильтрация → очистка (end-to-end)
│   ├── collect_data.py              # Вызов HHParser → data/raw/*.json
│   ├── filter_data.py               # Фильтр ролей аналитиков
│   ├── train_models.py              # Обучение SalaryPredictor + ARIMA
│   └── merde_databases.py           # [ОПЕЧАТКА: должно быть merge_databases.py]
├── docker/
│   ├── web.Dockerfile               # Streamlit-контейнер
│   └── parser.Dockerfile            # Контейнер парсера (без портов)
├── models/
│   ├── salary_predictor.pkl         # Обученный SalaryPredictor (pickle)
│   ├── arima_model.joblib           # SARIMA-модель
│   ├── arima_metrics.json           # AIC, BIC, MAE SARIMA
│   └── salary_metrics.json          # R², MAE, RMSE SalaryPredictor
├── finaldata/                       # Очищенные CSV (общий bind-mount)
├── data/raw/                        # Сырые JSON (приватный volume парсера)
├── tests/                           # pytest / unittest
├── docker-compose.yml
├── requirements.txt
└── run.py                           # Запуск Streamlit (совместимость с PyInstaller)
```

---

## 4. Поток данных

```
hh.ru API
   │  HHParser.fetch_vacancies()
   │  Фильтр analyst_role_ids: [10, 134, 148, 150, 156, 163, 164]
   ▼
data/raw/*.json  (parser container, приватный volume)
   │  scripts/filter_data.py
   ▼
data/processed/*.json  (отфильтровано по ролям)
   │  DataCleaner.run_full_clean()
   ▼
finaldata/new_cleaned_vacancies_YYYYMMDD_HHMMSS.csv
   │  (bind-mount, виден web-контейнеру)
   │
   ├─► VacanciesRepository.load_latest()  →  HomeService  →  home.py
   ├─► ClusteringService.perform_clustering()  →  clusters.py
   ├─► TimeSeriesAnalyzer.train() / forecast()  →  arima.py
   └─► SalaryPredictor.train() / predict()  →  predict_salary.py
```

---

## 5. Инфраструктура (Docker Compose)

Два сервиса:

| Сервис | Dockerfile | Порты | Роль |
|--------|-----------|-------|------|
| `web` | `docker/web.Dockerfile` | 8501:8501 | Streamlit-дашборд (read-only finaldata) |
| `parser` | `docker/parser.Dockerfile` | — | Сбор + очистка данных (cron) |

**Volumes:**
- `./finaldata` — bind-mount (общий): парсер пишет, web читает
- `raw_data` — приватный volume парсера (raw JSON)
- `cache_data` — clustering cache (web)
- `models_data` — ML-модели (web)

**Инвалидация кэша:** при появлении нового CSV изменяется MD5-checksum
(`VacanciesRepository.compute_checksum()`), что сбрасывает `@st.cache_resource` сервисов.

---

## 6. ML-компоненты

### 6.1 SalaryPredictor (`src/ml/salary_predictor.py`)

Обучает три модели + ансамбль, выбирает лучшую по R² на тестовой выборке.

| Модель | Алгоритм | Особенности |
|--------|----------|-------------|
| RandomForest | 300 деревьев, depth=12 | Только числовые фичи + target encoding |
| CatBoost | 2000 iter, depth=7, early_stopping=100 | Нативные cat_features + text_features |
| GradientBoosting | 300 iter, depth=5 | Только числовые фичи + target encoding |
| Ансамбль | R²-взвешенное среднее трёх | Используется если его R² лучший |

**Целевая переменная:** `salary_avg` (среднее из from/to, или одно из них)  
**Требования ТЗ:** R² ≥ 0.7, MAE ≤ 20% от средней ЗП  
**Сохранение:** `models/salary_predictor.pkl` (pickle)

**FeatureEngineer (`src/data_processing/feature_engineering.py`):**
- Категориальные: ordinal encoding для RF/GB; нативные индексы для CatBoost
- Target Encoding (сглаженный): вычисляется ТОЛЬКО по train-выборке, fallback = mean(y_train)
- TF-IDF: skills_list (top-20), requirement + responsibility (top-30 биграм)
- Бинарные флаги: has_test, premium, response_letter_required, accept_labor_contract
- Теги грейда: is_senior / is_middle / is_junior (по названию вакансии)

### 6.2 ClusteringService (`src/services/clustering_service.py`)

Автоматически выбирает алгоритм по типу признаков:

| Признаки | Алгоритм | Метрика |
|----------|----------|---------|
| Только числовые | K-Means | Euclidean |
| Только категориальные | K-Modes (Cao init) | Hamming |
| Смешанные | K-Prototypes | Gower (precomputed) |

**Оптимизация K:** перебор k ∈ [k_min, k_max], выбор по максимуму Silhouette Score.

**Нейминг кластеров:**
- Профессия попадает в название, если занимает ≥ 30% вакансий кластера
- Тег зарплаты: 🟢 > 70-й перцентиль / 🟡 30–70-й / 🔴 < 30-й перцентиль рынка
- Топ-5 навыков из `skills_list`

**Кэш:** `ClusteringCache` (joblib), ключ = MD5(version | checksum | features | k_range)

### 6.3 TimeSeriesAnalyzer (`src/ml/arima_analyzer.py`)

SARIMA-модель на временном ряду количества вакансий по датам публикации.

- Частота: D (день) / W (неделя) / ME (месяц)
- Сезонный период m: 7 / 52 / 12 (определяется автоматически по частоте)
- Преобразование: log1p (для неотрицательности прогноза), обратно через expm1
- Проверка стационарности: ADF-тест (Dickey-Fuller)
- Доверительные интервалы: Monte Carlo (1000 путей в log-пространстве)
- Сохранение: `models/arima_model.joblib` + `models/arima_metrics.json`

---

## 7. Нетривиальные архитектурные решения

| Решение | Где | Обоснование |
|---------|-----|-------------|
| Target encoding только по train | `salary_predictor.py:_add_target_encoding()` | Предотвращение утечки данных (data leakage) |
| CatBoost с `text_features` | `salary_predictor.py:225–265` | Позволяет учитывать текст требований без ручного TF-IDF |
| Gower distance для K-Prototypes | `clustering_service.py:80` | Единственный вариант корректной метрики для смешанных данных |
| MD5-checksum для инвалидации кэша | `vacancies_repository.py:48` | Web-контейнер обнаруживает новый CSV без перезапуска |
| @st.cache_resource, ключ = checksum | `clusters.py:33`, `home.py:8` | Сервис пересоздаётся только при смене данных, не при каждом запросе |
| Ансамбль по R²-весам | `salary_predictor.py:298–313` | Более точные модели получают пропорционально больший вес |
| Два Docker-сервиса | `docker-compose.yml` | Парсер не имеет портов → изолирован от внешнего доступа |

---

## 8. Схема domain-моделей

```python
HomeStats(total_vacancies, with_salary, avg_salary, active_vacancies, last_updated)

ClusterEntity(id, title, description, vacancies_count, avg_salary,
              skills, remote_rate, median_salary, popular_regions, salary_rate)

ClusteringResult(method_name, n_clusters, silhouette_score, clusters, k_scores)

SalaryPredictionResult(predicted_salary, currency, confidence_interval,
                       market_comparison_chart)
```

---

## 9. Конфигурация окружения

Файл `.env` (обязателен):
```
HH_USER_AGENT=DataTrack/1.0 (contact@example.com)
LOG_LEVEL=INFO
```

---

## 10. Запуск

**Локально:**
```bash
pip install -r requirements.txt
python run.py
# или
streamlit run app/main.py
```

**Docker:**
```bash
docker-compose up --build
# web: http://localhost:8501
```

**Обучение моделей:**
```bash
python scripts/train_models.py --compare   # RF + CatBoost + GB + Ансамбль
```
