FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=development

EXPOSE 8000 5000

CMD ["bash", "-c", "uvicorn src.api_inference:app --host 0.0.0.0 --port 8000 --reload"]