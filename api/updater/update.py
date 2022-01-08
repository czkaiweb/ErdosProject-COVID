from sodapy import Socrata
import pandas as pd
import numpy as np
from pickle import dump, load
from statsmodels.tsa.api import ARIMA
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sys import argv
from os import listdir, remove
import datetime
import warnings
from multiprocessing import Pool
import json
from shutil import move

dt = datetime

_duration = 14
_default_date_format = "%Y-%m-%d"

def next_day(date):
    return int_to_date(date_to_int(date) + 1)

def add_row(df):
    """ Add a row of nans, in place"""
    date = df.date.iloc[-1]
    next_date = next_day(date)
    df.at[df.index.max()+1] = [np.nan for _ in range(len(df.columns))]
    df.at[df.index.max(),"date"] = next_date
    return df
    

def generate_arima(df, date=None, write=False):
    N = df.index.max()
    if date is None:
        while np.isnan(df.beds.loc[N]):
            N -= 1
        if date is None:
            date = df.date.loc[N]
    else:
        while df.date.loc[N] != date:
            N -= 1
    if np.any(np.isnan(df.beds.loc[50:N])):
        return None
    with warnings.catch_warnings():
        success = False
        n = 50
        while not success:
            try:
                warnings.filterwarnings("ignore")
                model = ARIMA(df.loc[n:N].beds, order = (5,2,2))
                model_fit = model.fit()
                predicted = model_fit.forecast(14)
                if predicted.max() > df.iloc[N].beds * 4: # Guess: numerical error
                    raise ValueError()
                success = True
            except:
                # ARIMA.fit sometimes has difficulty with L/U
                # decomposition. If this happens, adjust the training
                # window and try again.
                n += 1
                if n > 150:
                    print("Uh oh")
                    raise ValueError()
            
    if write:
        for n in range(1,15):
            M = N + n
            if M not in df.index:
                add_row(df)
                assert M in df.index,df
            df.at[M, f"arima_{n}"] = predicted.iloc[n-1]
    return predicted

def update_state(s):
    """ s = covid_state[state], covid_state_old[state]
    operates on covid_state_old in place
    """
    df, df_old = s
    last_index = df_old.index.max()
    while np.isnan(df_old.beds.loc[last_index]):
        last_index -= 1
    
    N = df.index.max()
    last_date = df_old.date.loc[last_index]
    while df.date.loc[N] > last_date:
        N -= 1
    N += 1
    
    M = last_index + 1
    while N <= df.index.max():
        if M not in df_old.index:
            add_row(df_old)
            assert M in df_old.index
        for column in ["beds", "cases_7day"]:
            df_old.at[M, column] = df[column].loc[N]
        generate_arima(df_old, date=df.date.loc[N], write=True)
        N += 1
        M += 1
    return df_old

def write_range(s):
    state, df, start_date, end_date = s
    date = start_date
    while date <= end_date:
        generate_arima(df, date=date, write=True)
        date = next_day(date)
    return df, state

def date_to_int(date_string: str, form: str=_default_date_format) -> int:
    """Return date date_string in format form as an integer"""
    return dt.datetime.strptime(date_string, form).toordinal()

def int_to_date(ordinal: int, form: str=_default_date_format) -> str:
    """Return the day number ordinal to as a string, formatted with form"""
    return dt.datetime.fromordinal(ordinal).strftime(form)

# If the script is run with no arguments, then download data from CDC.
# If there is an argument, don't download fresh data but instead load
# in saved data (will still refit model and regenereate images).
#
# Socrata has an API for accessing CDC (and other) data sets
# I registered our project for an App Token: YeLTGbOCV2gjoCenq2ve5LzDm,
# which should be included in our requests to Socrata API.
# The datasets to download are:
# Vaccinations, identifier unsk-b7fc [eliminated]
# Daily cases, identifier 9mfq-cb36
# Hospital utilization, identifier g62h-syeh

print("Downloading data from the CDC")
client = Socrata("data.cdc.gov",
                "YeLTGbOCV2gjoCenq2ve5LzDm")
dataframe_list = [] # Downloaded dataframes go into this list
for record_identifier in ["9mfq-cb36"]:#, "unsk-b7fc"]:
    results = client.get(record_identifier, limit=50000)
    df = pd.DataFrame.from_records(results)
    dataframe_list.append(df)

client = Socrata("healthdata.gov",
                "YeLTGbOCV2gjoCenq2ve5LzDm")
results = client.get("g62h-syeh", limit=50000)
df = pd.DataFrame.from_records(results)
dataframe_list.append(df)

print("Preprocessing")
# We only want a few columns from each dataset
pruned_df_list = []

#     df = dataframe_list[0]
#     df_pruned = df[["date", "location","administered"]]
#     pruned_df_list.append(df_pruned.rename({"location":"state",
#                                            "administered":"vacc"}, axis=1))

df = dataframe_list[0]
df_pruned = df[["submission_date","state","new_case"]]
pruned_df_list.append(df_pruned.rename({"submission_date":"date",
                                       "new_case":"cases"}, axis=1))

df = dataframe_list[1]
df_pruned = df[["date","state","inpatient_beds_used_covid"]]
pruned_df_list.append(df_pruned.rename({"inpatient_beds_used_covid":"beds"},
                                       axis=1))

# Include only date, not time, in date column
for df in pruned_df_list:
    for n in df.date.index:
        df.date[n] = df.date[n].split("T")[0]

# Start data on 2020-04-01, end datasets at the earliest
# end date of the three datasets (they are sometimes off by one)
start_date = "2020-04-01"
end_date = min(df.date.max() for df in pruned_df_list)

# 50 states plus Washington DC and Puerto Rico
state_list = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
              'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
              'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
              'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR',
              'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI',
              'WV', 'WY']

# Separate data by state
covid_state = {}
for state in state_list:
    for n in [1,0]:
        df = pruned_df_list[n]
        df_state = df.loc[df.state == state].copy()
        df_state.sort_values("date", inplace=True)
        df_state = df_state.loc[(df_state.date>=start_date) & \
                               (df_state.date<=end_date)].copy()
        if n == 1:
            df_all = df_state[["date"]].reset_index(drop=True).copy()
            df_all["beds"] = df_state["beds"].values.astype(np.float64)
        elif n == 0:
            df_all["cases"] = df_state.cases.values.astype(np.float64)

    L = []
    for n in range(len(df_all)):
        if n < 6:
            L.append(np.mean(df_all.cases.loc[:n]))
        else:
            L.append(np.mean(df_all.cases.loc[n-6:n]))
    df_all["cases_7day"] = L

    covid_state[state] = df_all[["date","beds","cases_7day"]].copy()

print("Reading existing data")
# Read in existing data
covid_state_old = {}
def read_old():
    covid_state_old = {}
    for filename in listdir("covid_forecaster_volume/data"):
        if filename[-3:] == ".gz":
            state = filename[:-3]
            assert state in state_list
            covid_state_old[state] = pd.read_csv(f"covid_forecaster_volume/data/{filename}",
                                                compression='gzip')
    assert len(covid_state_old) == 52, len(covid_state_old)
    return covid_state_old

covid_state_old = read_old()

print("Updating (incl. ARIMA predictions)")
# Update old data and write it
for state in state_list:
    update_state((covid_state[state], covid_state_old[state]))

    name_new = f"covid_forecaster_volume/data/{state}.gz"
    #name_old = f"covid_forecaster_volume/data/{state}.gz.bak"
    #try:
    #    remove(name_old)
    #except:
    #    pass
    #move(name_new, name_old)
    covid_state_old[state].to_csv(name_new,
                                 index=False,
                                 compression='gzip')
covid_state = covid_state_old

print("Adding lag")
# Add lag variables
for state in state_list:
    for s in ["beds","cases_7day",]:
        for f in range(1,_duration+1):
            covid_state[state][f"{s}_{f}"] = \
                covid_state[state][s].shift(f)
    covid_state[state] = covid_state[state].copy()
    
# Train the model
def get_predictions(state, date=None):
    df = covid_state[state]
    if date is None:
        N = df.index.max()-_duration
    else:
        N = df.index.max()
        while df.date.loc[N] > date:
            N -= 1
    df_tuned = df.loc[N-25: N]
    pred_list = []
    for future in range(1, _duration+1):
        reg = Pipeline([('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=10))])
        reg.fit(df_tuned[[f"beds_{future}",
                    f"cases_7day_{future}",
                   f"arima_{future}"]].values,
               df_tuned.beds)
        df_to_predict = df.loc[N+future]
        arima_prediction = df_to_predict[f"arima_{future}"]
        linear_prediction = reg.predict(np.concatenate((df_to_predict[[f"beds_{future}",
                                                                       f"cases_7day_{future}"]],
                              [arima_prediction])).reshape(1,-1))[0]
        baseline_prediction = df_to_predict[f"beds_{future}"]
        prediction = np.mean([arima_prediction,
                             linear_prediction,
                             baseline_prediction])
        pred_list.append(prediction)
    return pred_list

print("Making predictions")
pred = {}
for state in state_list:
    try:
        pred[state] = get_predictions(state)
    except:
        pred[state] = None
        print("Prediction error:", state)
pred_historic = {}
for state in state_list:
    date = covid_state[state].date.iloc[-1]
    new_date = int_to_date(date_to_int(date) - 2*_duration)
    try:
        pred_historic[state] = get_predictions(state, date=new_date)
    except:
        pred_historic[state] = None

# Function for generating graphs with predictions
def show_prediction(state, save=False,
                   historic=False):
    df = covid_state[state]
    N = df.index.max()
    while np.isnan(df.beds.loc[N]):
        N -= 1
    if historic:
        actual = df.beds.loc[N-35:N]
        predicted = np.concatenate(([actual.loc[N-14]],pred_historic[state]))
        predicted_x = range(len(actual)-15,len(actual)+len(predicted)-15)
    else:
        actual = df.beds.loc[N-21:N]
        predicted = np.concatenate(([actual.loc[N]],pred[state]))
        predicted_x = range(len(actual)-1,len(actual)+len(predicted)-1)
    actual_x = range(len(actual))
    
    
    
    plt.figure(figsize=(12,9))
    plt.plot(actual_x,actual,
             label="Actual" if historic else "Observed",
            c="blue")
    plt.plot(predicted_x,
             predicted,
             '--',
             label="Predicted",
            c="red")
    x = len(actual)- (15 if historic else 1)
    y = actual.iloc[-15 if historic else -1]
    plt.plot([x],
             [y],'o',
             c="black",markersize=8)
    if not historic:
        plt.text(len(actual)+.4,actual.iloc[-1]*0.99,
                 str(int(actual.iloc[-1])),
                fontsize=16,
                color="black")
    plt.legend()
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Beds in use", fontsize=16)

    # Add a few dates as labels to x-axis
    x_pos = []
    x_date = []
    end_date = covid_state[state].date.loc[N]
    if historic:
        a,b = -5,1
    else:
        a,b = -3,3
    for k in range(a,b):  
        new_date = datetime.datetime.strftime(datetime.datetime.strptime(end_date,
                                                                         "%Y-%m-%d")\
                                              + datetime.timedelta(days=7*k),
                                                          "%Y-%m-%d")
        new_pos = len(actual)-1+7*k
        if k == 0 and not historic:
            new_date += "\n(most recent\nobservation)"
        x_pos.append(new_pos)
        x_date.append(new_date)
    plt.xticks(x_pos, x_date, fontsize=12)
    title_string = f"COVID hospitalizations in {state}"
    if historic:
        title_string += "\n(two week prior forecast)"
    plt.title(title_string, fontsize=22)
    if save:
        if historic:
            plt.savefig(f"covid_forecaster_volume/graphs/previous/{state}.png")
        else:
            plt.savefig(f"covid_forecaster_volume/graphs/{state}.png")
    else:
        plt.show()
    plt.close()

print("Generating graphs and saving predictions")
# Save predictions (graphical and numerical)
for state in state_list:
    if pred[state] is not None:
        show_prediction(state, save=True)
    if pred_historic[state] is not None:
        show_prediction(state, historic=True, save=True)
    if pred[state] is None:
        continue
    df = covid_state[state]
    date_list = list(df.date.iloc[-_duration:])
    
    D = {date_list[n]:int(np.round(pred[state][n],0)) for n in range(_duration)}
    with open(f"covid_forecaster_volume/forecasts/{state}", 'w+') as file:
        file.write(json.dumps(D))
        
