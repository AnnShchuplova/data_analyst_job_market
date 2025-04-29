# DataTrack — аналитический дашборд рынка труда

Веб-сервис для исследования рынка труда специалистов по анализу данных на основе данных **hh.ru**.
Охватывает полный цикл: сбор → очистка → хранение → интерактивный дашборд с ML.

Курсовой проект · НИУ ВШЭ, ФКН, БПИ247 · Тищенко Н.А. · 2026

---

## Возможности

| Страница | Что делает |
|---|---|
| 🏠 Главная | Сводка по базе: число вакансий, доля с ЗП, средняя ЗП, дата обновления |
| 📊 Агрегация | Фильтрация по региону / опыту / ЗП / роли / навыкам. 6 вкладок визуализации. Вкладка «Мой профиль» — перцентиль пользователя на рынке + рекомендации навыков. Экспорт в CSV |
| 🧩 Кластеры | Группировка вакансий по выбранным признакам. Алгоритм подбирается автоматически: K-Means / K-Modes / K-Prototypes. Оптимальное K — по Silhouette Score |
| 📈 Тренды | Динамика числа вакансий и зарплат. SARIMA-прогноз с доверительными интервалами (Monte Carlo) |
| 💰 Прогноз ЗП | Оценка зарплаты по профилю. Ансамбль: RandomForest + CatBoost + GradientBoosting. Обучение прямо из интерфейса |

---

## Быстрый старт

### Docker (рекомендуется)

```bash
cp .env.example .env          # заполни HH_USER_AGENT
docker compose up --build
```

Дашборд: **http://localhost:8501**

При первом запуске данных нет — нужно запустить парсер или положить CSV в `finaldata/`.

### Локально

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

---

## Архитектура

```
┌─────────────────────────────────────┐
│           app/views/                │  Streamlit UI
│  home · aggregation · clusters      │
│  arima · predict_salary             │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│           src/services/             │  Бизнес-логика
│  HomeService · ClusteringService    │
│  TimeSeriesAnalyzer · SalaryPredictor│
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│        src/infrastructure/          │  Данные
│  VacanciesRepository (CSV + MD5)    │
│  ClusteringCache (joblib)           │
└────────────────┬────────────────────┘
                 │
         finaldata/*.csv
```

**Инвалидация кэша:** `VacanciesRepository` вычисляет MD5 последнего CSV каждые 5 минут.
При смене файла `@st.cache_resource`-сервисы пересоздаются автоматически — без перезапуска контейнера.

---

## Docker Compose

Два независимых сервиса:

```
web     — Streamlit дашборд (порт 8501, read-only finaldata)
parser  — сборщик hh.ru + cleaner (без портов, cron)
```

**Тома:**
- `./finaldata` — bind-mount: парсер пишет, web читает
- `./models` — bind-mount: модели обучаются внутри контейнера и сохраняются на хост
- `raw_data` — приватный volume парсера (сырые JSON)
- `cache_data` — кэш кластеризации (web)

---

## ML-компоненты

### Прогноз зарплаты

Три модели + R²-взвешенный ансамбль. Обучение через вкладку «⚙️ Обучение модели» или скриптом:

```bash
python scripts/train_models.py
```

| Модель | Детали |
|---|---|
| RandomForest | 300 деревьев, depth=12, target encoding + TF-IDF |
| CatBoost | 2000 итераций, depth=7, нативные cat/text features |
| GradientBoosting | 300 итераций, depth=5 |
| Ансамбль | R²-взвешенное среднее, используется если его R² выше остальных |

Цели ТЗ: **R² ≥ 0.7**, **MAE ≤ 20%** от средней ЗП.

Модель сохраняется в `models/salary_predictor.pkl` + `models/salary_predictor.cbm` (нативный формат CatBoost, независим от версии numpy).

### Кластеризация

Автоматический выбор алгоритма по типу выбранных признаков:

| Признаки | Алгоритм | Метрика расстояния |
|---|---|---|
| Числовые | K-Means | Евклидово |
| Категориальные | K-Modes (Cao init) | Хэмминга |
| Смешанные | K-Prototypes | Говера (Gower) |

Перебор k от `k_min` до `k_max`, выбор по максимуму Silhouette Score.
Результат кэшируется в `data/cache/` (ключ = MD5 данных + набор признаков + диапазон k).

### Временные ряды

SARIMA на динамике числа вакансий. Частота задаётся пользователем (день / неделя / месяц).
Преобразование log1p обеспечивает неотрицательность прогноза. Доверительные интервалы — Monte Carlo (1000 путей).

---

## Структура проекта

```
├── app/
│   ├── main.py                     # Точка входа, роутер страниц
│   └── views/                      # Страницы Streamlit
├── src/
│   ├── domain/models.py            # Датаклассы (HomeStats, ClusterEntity, ...)
│   ├── infrastructure/             # VacanciesRepository, ClusteringCache
│   ├── services/                   # Бизнес-логика
│   ├── data_collection/            # hh.ru API-клиент
│   ├── data_processing/            # DataCleaner, FeatureEngineer
│   └── ml/                         # SalaryPredictor, TimeSeriesAnalyzer
├── scripts/
│   ├── run_pipeline.py             # Сбор → очистка → финальный CSV (end-to-end)
│   └── train_models.py             # Обучение SalaryPredictor + ARIMA
├── docker/
│   ├── web.Dockerfile
│   └── parser.Dockerfile
├── models/                         # Обученные модели (bind-mount в Docker)
├── finaldata/                      # Очищенные CSV (bind-mount, общий)
├── docker-compose.yml
├── requirements-web.txt            # Зависимости web-контейнера
├── requirements-parser.txt         # Зависимости parser-контейнера
└── .env.example
```

---

## Конфигурация

Файл `.env` (скопируй из `.env.example`):

```env
HH_USER_AGENT=DataTrack/1.0 (your@email.com)
LOG_LEVEL=INFO
```

---

## Стек

Python 3.11 · Streamlit 1.38 · CatBoost 1.2.10 · scikit-learn 1.7 · pandas · numpy 2.2 · statsmodels · kmodes · gower · Plotly · Docker Compose
