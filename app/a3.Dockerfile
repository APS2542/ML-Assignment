FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY code /app/code
EXPOSE 8000
CMD ["gunicorn","-w","2","-k","gthread","--threads","4","-t","120","code.app:server","--bind","0.0.0.0:8000"]
