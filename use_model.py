import pickle 
import numpy as np 

local_classifier = pickle.load(open('classifier.pickle', 'rb'))
local_scaler = pickle.load(open('scaler.pickle', 'rb')) 

new_obs = np.array([[40, 20000], [50,50000], [23, 15000]])

new_preds = local_classifier.predict(local_scaler.transform(new_obs))
new_probs = local_classifier.predict_proba(local_scaler.transform(new_obs)) 

print(new_preds)
print(new_probs)
