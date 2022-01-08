FROM python:3.9-slim-bullseye


RUN useradd -m forecaster
VOLUME /home/forecaster/covid_forecaster_volume
RUN chown -R forecaster:forecaster /home/forecaster/covid_forecaster_volume

USER forecaster
WORKDIR /home/forecaster/


COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY update.py .

#CMD whoami && id -u $(whoami) && ls -l && touch covid_forecaster_volume/tester
CMD /usr/local/bin/python3.9 update.py
#CMD cat covid_forecaster_volume/forecasts/AK
#CMD ls -l /home/forecaster && whoami
