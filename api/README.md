The COVID-19 forecasting tool has several components:
- A containerized application that fetches fresh data from the CDC,
    retrains our models, and generates new predictions and graphs.
- A containerized application to service the API, using flask and gunicorn
- A webpage, with links to our predictions (present and past)

Installation instructions for an Ubuntu system. From the `api` directory:
- Dockerfiles are included in this repository, but you will need to build
    the images yourself. First, make sure you have docker installed. If not,
    you can install with  
    `sudo snap install docker` .
- Build the docker image for the updater:  
`sudo docker build updater -t covid_updater`.
- Build the docker image for the API:  
`sudo docker build forecaster -t covid_forecaster`.
- Install a crontab to run the updater daily:  
`sudo cp covid_update /etc/cron.d/`
- Install a systemd service to operate the api:  
`sudo cp covid.service /etc/systemd/system/`  
`sudo systemctl enable covid.service`  
`sudo systemctl start covid.service`
- Add Apache configuration file (Apache2 required, install with `sudo apt install apache2`):  
`sudo cp covid.conf /etc/apache2/sites-available/`  
`sudo a2ensite covid`  
`sudo systemctl reload apache2`
