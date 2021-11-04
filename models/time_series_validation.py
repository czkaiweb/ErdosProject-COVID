import pandas as pd
import numpy as np
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.base import clone
import datetime as dt

_default_date_format = "%Y-%m-%d"

def date_to_int(date_string: str, form: str=_default_date_format) -> int:
    """Return date date_string in format form as an integer"""
    return dt.datetime.strptime(date_string, form).toordinal()

def int_to_date(ordinal: int, form: str=_default_date_format) -> str:
    """Return the day number ordinal to as a string, formatted with form"""
    return dt.datetime.fromordinal(ordinal).strftime(form)

def add_lag(df, feature_dict):
    """Return a dataframe obtained from df by adding lag
    feature called featurename-n for each featurename in feature_dict
    and each n in feature_dict[featurename]"""
    series_list = []
    for feature in feature_dict.keys():
        for n in feature_dict[feature]:
            if n != 0:
                series = df[feature].shift(n).copy().rename(f"{feature}-{n}")
                series_list.append(series) 
    untrimmed = pd.concat([df[feature_dict.keys()]] + series_list, axis=1).copy()
    return untrimmed.iloc[max(np.max(feature_dict[feature]) \
                                  for feature in feature_dict \
                                 if len(feature_dict[feature]) > 0):]

def load_states():
    """Read state covid data"""
    covid_state = {}
    state_list = []
    state_dir = join("..", "data", "input", "simple_states")
    for filename in listdir(state_dir):
        if filename[-4:] == ".csv":
            state = filename[:-4]
            covid_state[state] = pd.read_csv(join(state_dir, filename))
            state_list.append(state)
    return covid_state, state_list
    
class FuturePrediction:
    """Time series cross-validator
    
            Provides train/test indices to split data in train/test sets.

    Parameters
    ----------
    train_length : int
        Length of training period.
    
    future_time : int
        Number of time periods in the future we want to predict
        
    interval : int, default=1
        One out of this many time periods is chosen to validate
    
    randomize : bool, default=False
        Randomize which days are chosen to validate
    """
    def __init__(self, train_length, future_interval,
                 interval=1, randomize=False):
        self.train_length = train_length
        self.future_interval = future_interval
        self.interval = interval
        self.randomize = randomize
        
    def split(self, df):
        length = len(df)
        index = 0
        max_index = length - self.train_length - self.future_interval
        while index < max_index:
            yield np.arange(index, index+self.train_length), \
                    np.array([index+self.train_length+self.future_interval-1])
            if self.randomize:
                index += np.randint(1, 2*self.interval)
            else:
                index += self.interval
        
def plot_predictions(reg, df, train_length, future_interval, features):
    actuals, baselines, predicteds = list(), list(), list()
    cv = FuturePrediction(train_length, future_interval)
    for train_index, test_index in cv.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        actual = test.beds.iloc[0]
        reg_copy = clone(reg)
        reg_copy.fit(train[features], train["beds"])
        predicted = reg_copy.predict(test[features])[0]
        target = f"beds-{future_interval}"
        baseline = test[target].iloc[0]
        actuals.append(actual)
        baselines.append(baseline)
        predicteds.append(predicted)
    
    plt.plot(actuals, label="actual")
    plt.plot(baselines, label="baseline")
    plt.plot(predicteds, label="predicted")
    start_date = df.date.iloc[train_length+future_interval-1]
    end_date = df.date.iloc[-1]
    plt.xticks([0,len(actuals)-1], [start_date, end_date])
    plt.xlabel("Date")
    plt.ylabel("Covid beds in use")
    plt.legend()
    return None

def validate(covid_state, regressor, train_length,
             future_interval, max_lag, interval, param_grid,
            plot=False):
    """ Performs state-by-state optimization of hyperparameters
    
        Parameters:
        covid_state: dictionary
            Dictionary of data frames with state data
            
        regreessor: regression object
        train_length: integer
            Number of training data points to fit
        future_interval: integer
            Number of days in the future we want to predict
        max_lag: integer
            Max number of days worth of data to use in each data point
        interval: integer
            Space between testing days for GridSearchCV
        param_grid: dictionary
            Hyperparameter values to check (passed to GridSearchCV)
    """
    cv = FuturePrediction(train_length, future_interval, interval)
    cv_test = FuturePrediction(train_length, future_interval)
    ratios = []
    for state in covid_state:
        df = add_lag(covid_state[state],
                    {"beds": range(future_interval, future_interval+max_lag),
                    "cases_7day": range(future_interval, future_interval+max_lag),
                    "vaccines": [future_interval],
                    "date":[],
                    "day_number":[]})
        train = df.loc[(df.day_number >= date_to_int("2021-01-01")) &
                      (df.day_number <= date_to_int("2021-07-01"))].copy()
        test = df.loc[(df.day_number > date_to_int("2021-07-01")) &
                      (df.day_number < date_to_int("2021-09-01"))].copy()

        gs = GridSearchCV(regressor,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=cv.split(train),
                    n_jobs=-1)
        
        features = [f"beds-{k}" for k in range(future_interval, future_interval+max_lag)] + \
                    [f"cases_7day-{k}" for k in range(future_interval, future_interval+max_lag)] + \
                    [f"vaccines-{future_interval}"]
        gs.fit(train[features], train["beds"])
        reg = gs.best_estimator_
        q = cross_validate(reg, test[features],
                       test["beds"],
                       scoring="neg_mean_squared_error",
                       cv=cv_test.split(test))
        regression_mse = -q["test_score"].mean()
        baseline_mse = ((test[f"beds-{future_interval}"].iloc[train_length+future_interval-1:]-\
                         test["beds"].iloc[train_length+future_interval-1:])**2).mean()
        ratio = regression_mse/baseline_mse
        print(state, ratio)
        #print(gs.best_params_)
        ratios.append(ratio)
        if plot:
            plot_predictions(reg, test, train_length, future_interval, features)
            plt.title(f"{state} holdout data")
            plt.show()
    return np.mean(ratios)