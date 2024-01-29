FROM python:3.9
 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]