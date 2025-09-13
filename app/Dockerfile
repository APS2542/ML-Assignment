FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# (ตัวช่วยถ้าบาง pkg ยังอยากคอมไพล์เอง)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# อัปเกรด pip/setuptools/wheel
RUN python -m pip install --upgrade pip setuptools wheel

# ติดตั้งจาก requirements โดย "รับเฉพาะ wheel"
COPY requirements.txt .
RUN pip install --only-binary=:all: --no-cache-dir -r requirements.txt

# คัดลอกซอร์สทั้งหมด
COPY . .

EXPOSE 80
# ปรับ path ให้ตรงกับของคุณ (ในรูปก่อนหน้า app.py อยู่ที่ code/pages/app.py)
CMD ["python", "code/pages/app.py"]


