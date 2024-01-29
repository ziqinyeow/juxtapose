FROM python:3.9
 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY . .


RUN pip install --upgrade pip && pip install --no-cache-dir .