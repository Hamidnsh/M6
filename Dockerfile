FROM python:3.8-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN  apt-get update
RUN apt-get install libgomp1

COPY . .
