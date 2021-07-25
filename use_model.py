import pickle 
import numpy as np 
import json
import requests 

url = "http://127.0.0.1:8005/model"
new_obs = np.array([[40, 20000], [50,50000], [23, 15000]])

request_data = json.dumps({'age': 50, 'salary': 10000})
response = requests.post(url, request_data)
print(response.text)


