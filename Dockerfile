FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src

COPY src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
