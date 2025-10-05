FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app /app/app
COPY src /app/src

EXPOSE 8000
ENV MLFLOW_TRACKING_URI="" MODEL_URI=""
CMD ["gunicorn","-w","2","-k","gthread","--threads","4","-t","120","app.code.app:server","--bind","0.0.0.0:8000"]
