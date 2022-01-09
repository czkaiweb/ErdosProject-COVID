FROM python:3.9-slim-bullseye

RUN useradd -m forecaster
USER forecaster
WORKDIR /home/forecaster/

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir tmp
RUN mkdir tmp/graphs
RUN mkdir tmp/graphs/previous
RUN mkdir tmp/data
RUN mkdir tmp/forecasts

COPY update.py .
COPY copyover.sh .
RUN mkdir covid_forecaster_volume
VOLUME /home/forecaster/covid_forecaster_volume

USER root
WORKDIR /home/forecaster
CMD runuser -l forecaster -c "/usr/local/bin/python3.9 update.py" && /bin/bash copyover.sh
