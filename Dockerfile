FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY main.py /app/main.py
COPY backend/__init__.py /app/backend/__init__.py
COPY backend/app.py /app/backend/app.py
COPY frontend /app/frontend
RUN mkdir -p /app/backend/data

ENV PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
