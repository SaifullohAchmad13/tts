FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["python","tts_server.py"]
