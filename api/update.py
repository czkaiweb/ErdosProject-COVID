from sodapy import Socrata
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from pickle import dump, load
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sys import argv
from os import listdir
import datetime

# Should we set aside 80 days' worth of data?
holdout = True

if len(argv) == 1:
    # Download data from CDC
    print("Downloading data from the CDC")
    client = Socrata("data.cdc.gov",
                    "YeLTGbOCV2gjoCenq2ve5LzDm")
    dataframe_list = []
    for record_identifier in ["unsk-b7fc", "9mfq-cb36"]:
        results = client.get(record_identifier, limit=50000)
        df = pd.DataFrame.from_records(results)
        dataframe_list.append(df)

    client = Socrata("healthdata.gov",
                    "YeLTGbOCV2gjoCenq2ve5LzDm")
    results = client.get("g62h-syeh", limit=50000)
    df = pd.DataFrame.from_records(results)
    dataframe_list.append(df)


    # Some data cleaning
    pruned_df_list = []

    df = dataframe_list[0]
    df_pruned = df[["date", "location","administered"]]
    pruned_df_list.append(df_pruned.rename({"location":"state",
                                           "administered":"vacc"}, axis=1))

    df = dataframe_list[1]
    df_pruned = df[["submission_date","state","new_case"]]
    pruned_df_list.append(df_pruned.rename({"submission_date":"date",
                                           "new_case":"cases"}, axis=1))

    df = dataframe_list[2]
    df_pruned = df[["date","state","inpatient_beds_used_covid"]]
    pruned_df_list.append(df_pruned.rename({"inpatient_beds_used_covid":"beds"},
                                           axis=1))

    for df in pruned_df_list:
        for n in df.date.index:
            df.date[n] = df.date[n].split("T")[0]

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
        for n in [2,1,0]:
            df = pruned_df_list[n]
            df_state = df.loc[df.state == state].copy()
            df_state.sort_values("date", inplace=True)
            df_state = df_state.loc[(df_state.date>=start_date) & \
                                   (df_state.date<=end_date)].copy()
            if n == 2:
                df_all = df_state[["date"]].reset_index(drop=True).copy()
                df_all["beds"] = df_state["beds"].values.astype(np.float)
            elif n == 1:
                df_all["cases"] = df_state.cases.values.astype(np.float)
            else:
                admin = np.array(df_state.vacc).astype(np.float)
                df_all["vacc"] = np.concatenate((np.zeros(len(df_all)-len(admin)),
                                                 admin))
        L = []
        for n in range(len(df_all)):
            if n < 6:
                L.append(np.mean(df_all.cases.loc[:n]))
            else:
                L.append(np.mean(df_all.cases.loc[n-6:n]))
        df_all["cases_7day"] = L

        covid_state[state] = df_all
        df_all.to_csv(f"covid_data/{state}.csv")
else:
    state_list = []
    covid_state = {}
    for filename in listdir("covid_data"):
        if filename[-4:] == ".csv":
            state = filename[:-4]
            state_list.append(state)
            covid_state[state] = pd.read_csv(f"covid_data/{filename}")
    

# New data frame where columns are states
start_date = "2020-04-01"
    
if holdout:
    df = pd.DataFrame(np.array([covid_state[state].beds.iloc[:-80] for \
                            state in state_list]).transpose(),
                    columns=state_list,
                    index=covid_state["MA"].date.iloc[:-80])
else:
    df = pd.DataFrame(np.array([covid_state[state].beds for \
                            state in state_list]).transpose(),
                    columns=state_list,
                    index=covid_state["MA"].date)
start_date = "2020-04-01"
end_date = df["MA"].index.max()
df_diff = df.copy()
for state in state_list:
    # Compute backward differences (to make time series stationary)
    df_diff[state] = df[state].diff(1)
df_diff = df_diff.iloc[1:].copy()

# Fit the model
lag_order = 4 # What does this do?
future = 30
model = VAR(df_diff.values)
results = model.fit(lag_order)
with open("model.pickle", 'wb') as f:
    dump((results,
          df_diff.values[-lag_order:],
         df.iloc[-1],
         state_list,
         end_date),f)

predictions = results.forecast(df_diff.values[-lag_order:], future)

# Generate graphs
for state_num in range(len(state_list)):
    state = state_list[state_num]
    state_info = df[state]

    # Un-difference the sequence
    predicted = np.cumsum(predictions[:,state_num]) + state_info.iloc[-1]
    predicted = np.concatenate(([state_info.iloc[-1]],predicted))

    num_actual = len(state_info)
    num_actual = 60
    plt.figure(figsize=(8,8))
    plt.plot(range(num_actual),state_info.iloc[-num_actual:], label="Observed")
    plt.plot(range(num_actual-1, num_actual+future),
             predicted,
             '--',
             label="Predicted",
            c="red")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Beds used")
    x_pos = []
    x_date = []
    for k in range(-8,5,2):  
        new_date = datetime.datetime.strftime(datetime.datetime.strptime(end_date,
                                                                         "%Y-%m-%d")\
                                              + datetime.timedelta(days=7*k),
                                                          "%Y-%m-%d")
        new_pos = num_actual-1+7*k
        x_pos.append(new_pos)
        x_date.append(new_date)
    plt.xticks(x_pos, x_date)
    title_string = f"COVID hospital patients in {state}"
    if holdout:
        title_string += "\n(80 day data holdout)"
    plt.title(title_string)
    plt.savefig(f"graphs/{state}.png")
    plt.close()