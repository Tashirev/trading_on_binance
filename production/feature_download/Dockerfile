FROM python:3.10.11-alpine3.18

LABEL maintainer="tashirevd@mail.ru"

ENV TZ="Europe/Moscow"

WORKDIR /app

COPY . /app

WORKDIR /app/production/feature_download

RUN apk update && apk add bash

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main_features.py"]