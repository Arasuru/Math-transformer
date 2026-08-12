FROM python:3.11-slim

WORKDIR /app

COPY prod_requirements.txt .

RUN pip install --no-cache-dir -r prod_requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
