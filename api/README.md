The COVID-19 forecasting tool has several components:
- A containerized application that fetches fresh data from the CDC,
    retrains our models, and generates new predictions and graphs.
- A containerized application to service the API, using flask and gunicorn
- A webpage, with links to our predictions (present and past)

The suggested installation looks like the following
