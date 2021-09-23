# syntax=docker/dockerfile:1
FROM ubuntu:20.04
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apt update && \
    apt install -y python3.8 python3-pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run"]
