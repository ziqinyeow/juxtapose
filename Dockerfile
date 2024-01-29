FROM python:3.9
 
WORKDIR /app

COPY src/ .
COPY pyproject.toml .
COPY README.md .

RUN pip install --upgrade pip && pip install --no-cache-dir .