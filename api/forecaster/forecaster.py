from flask import Flask, request, jsonify
import json

json_header = {'content-type': 'application/json; charset=UTF-8'}
app = Flask(__name__)    

@app.route('/', methods=['GET'])
def get_prediction():
    
    args = request.args

    # check that all args are present
    desired_args = ['state']
    missing_args = [a for a in desired_args if args.get(a) is None]
    
    if len(missing_args) > 0:
        error_msg = f"argument(s) missing: {missing_args}"
        return (jsonify(error_msg), 422, json_header)
                
    
    # make predictions
    state = args.get("state").upper()
    try:
        with open(f"covid_forecaster_volume/forecasts/{state}") as file:
            D = json.loads(file.read())
    except:
        error_msg = f"Invalid state '{state}', please use two-letter postal abbreviation"
        return (jsonify(error_msg), 422, json_header)
     
    return (jsonify(D), 200, json_header)
    
if __name__ == '__main__':
    #app.run(host='0.0.0.0')
    app.run(use_reloader=True, debug=True)
