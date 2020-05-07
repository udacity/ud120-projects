FROM continuumio/miniconda3:latest

COPY requirements.txt .
RUN pip install -r requirements.txt
