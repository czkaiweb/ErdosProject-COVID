
# imports related to flask and loading the model
from flask import Flask, request, jsonify
import pickle
import datetime
import numpy as np

json_header = {'content-type': 'application/json; charset=UTF-8'}
app = Flask(__name__)    

@app.route('/', methods=['GET'])
def get_prediction():
    
    args = request.args

    # check that all args are present
    desired_args = ['state', 'duration']
    missing_args = [a for a in desired_args if args.get(a) is None]
    
    if len(missing_args) > 0:
        error_msg = 'argument(s) missing: {}'.format(missing_args)
        return (jsonify(error_msg), 422, json_header)
                
    # check that all args are floats
#     def arg_is_float(arg):
#         is_float = False
        
#         try:
#             x = float(arg)
#             is_float = True
#         except ValueError:
#             pass
        
#         return is_float
    
#     nonfloat_args = [a for a in desired_args if not arg_is_float(args.get(a))]
    
#     if len(nonfloat_args) > 0:
#         error_msg = 'argument(s) not float: {}'.format(nonfloat_args)
#         return (jsonify(error_msg), 422, json_header)
    
    # make predictions
    state = args.get("state")
    duration = args.get("duration")

    with open("model.pickle", "rb") as f:
        results, end_values, recent, state_list, date = pickle.load(f)
    
    # results,
    # end_values = df_diff.values[-lag_order:],
    # recent = df.iloc[-1]
    
    state_num = state_list.index(state)
    
    predictions = results.forecast(end_values, int(duration))
    predicted = np.cumsum(predictions[:,state_num]) + recent[state]
    
    D = {}
    D[f"{date} (observed)"] = recent[state]
    base = datetime.datetime.strptime(date,"%Y-%m-%d")
    for x in range(1, int(duration)+1):
        new_date = datetime.datetime.strftime(base + datetime.timedelta(days=x),
                                              "%Y-%m-%d")
        D[f"{new_date} (predicted)"] = predicted[x-1]    
    return (jsonify(D), 200, json_header)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
