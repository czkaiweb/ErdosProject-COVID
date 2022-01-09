#!/bin/bash

# Copy files written by update.py (this hack is necessary because we need to preserve ownership)
for DIR in data forecasts graphs graphs/previous
do
    cd /home/forecaster/tmp/$DIR
    for FILE in *
    do
        cat $FILE > /home/forecaster/covid_forecaster_volume/$DIR/$FILE
    done
done
