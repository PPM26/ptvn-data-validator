FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

EXPOSE 5500

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.fastapi.main:app", "--host", "0.0.0.0", "--port", "5500"]
