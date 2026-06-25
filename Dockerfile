FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    LLM_PROVIDER=litellm \
    LITELLM_BASE_URL=http://litellm:4000 \
    LITELLM_MODEL=qwen3:8b

WORKDIR ${APP_HOME}

RUN addgroup --system app \
    && adduser --system --ingroup app --home /home/app app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY clients ./clients
COPY models ./models
COPY services ./services
COPY main.py .

USER app

EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
