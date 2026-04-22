# Private parser/cleaner service. Never exposes a network port.
# Runs `scripts.run_pipeline` on a monthly cron schedule.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Moscow

# cron for scheduling; tzdata so crontab honours TZ.
RUN apt-get update && apt-get install -y --no-install-recommends cron tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-parser.txt .
RUN pip install --no-cache-dir -r requirements-parser.txt

# Parser needs data_collection + data_processing + scripts only.
COPY src/ ./src/
COPY scripts/ ./scripts/

# Register crontab.
COPY docker/parser-crontab /etc/cron.d/parser-cron
RUN chmod 0644 /etc/cron.d/parser-cron \
    && crontab /etc/cron.d/parser-cron \
    && touch /var/log/pipeline.log

RUN mkdir -p /app/data/raw /app/data/processed

# cron -f keeps PID 1 alive. tail the log so `docker logs` shows pipeline output.
CMD ["sh", "-c", "cron && tail -F /var/log/pipeline.log"]
