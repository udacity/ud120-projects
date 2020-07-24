FROM python:3.4

COPY requirements.txt .
RUN pip install -r requirements.txt
