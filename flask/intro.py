import pandas as pd
import numpy as np 
from scipy.cluster import hierarchy as hc



raw_data = pd.read_csv('./storepurchasedata.csv')
print(raw_data.describe())

print(raw_data.Purchased.value_counts()/40)
# check corrleations
print(raw_data.corr())

# create X and Y values 
dv = "Purchased"
X = raw_data.iloc[:, :-1].values
Y = raw_data.iloc[:, -1].values

# or 
ivs = [c for c in raw_data.columns if c != dv]
X_ = raw_data.loc[:, ivs].values
Y_ = raw_data.loc[:, dv].values

assert X.all() == X_.all()
assert Y.all() == Y_.all()

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 22)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print(X_train)
print(X_test)
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors = 3, metric = "minkowski", p =2)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
print(confusion_matrix(Y_test, y_pred))
p_s = precision_score(Y_test, y_pred)
r_s = recall_score(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
print(f"The precision is {p_s}")
print(f"The recall score is {r_s}")
print(f"Accuracy is {acc}")

new_prediction = classifier.predict(sc.transform(np.array([[40,50000]])))
new_prob =classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]
print(new_prediction)
print(new_prob)

# saving files 
import pickle
model_file = "classifier.pickle"
pickle.dump(classifier, open(model_file, "wb"))
scaler_file = "scaler.pickle"
pickle.dump(sc, open(scaler_file, "wb"))







