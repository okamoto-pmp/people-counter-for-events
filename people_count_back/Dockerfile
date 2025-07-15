FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY lambda/index.py .

EXPOSE 3000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=3000"]