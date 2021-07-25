from flask import Flask, request
import numpy as np
import pickle 
import sklearn

local_classifer = pickle.load(open('classifier.pickle', 'rb'))
local_sc = pickle.load(open('scaler.pickle', 'rb')) 

app = Flask(__name__)

@app.route('/model', methods = ['POST'])
def hello_world(): 
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    ob = np.array([[age, salary]])
    pred_prob = local_classifer.predict_proba(local_sc.transform(ob))[:,1]
    return f'The prediction is {pred_prob}'


if __name__ == '__main__':
    app.run(port=8005, debug=True)