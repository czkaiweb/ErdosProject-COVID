FROM python:3.9-slim-bullseye

RUN useradd -m forecaster
USER forecaster
WORKDIR /home/forecaster/

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY forecaster.py .
ENTRYPOINT ["/usr/local/bin/python3.9", "-m", "gunicorn", "-w", "2", "-b", ":5000", "-t", "360", "--reload", "forecaster:app"]
