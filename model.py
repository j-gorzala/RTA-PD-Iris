import numpy as np
import pandas as pd
import math
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from perceptron import Perceptron


iris=load_iris()

df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names']+['target'])
X = df[['sepal length (cm)', 'petal length (cm)']]
X = X.values
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model fitting
model = Perceptron(eta=0.00275, n_iter=1000)
model.fit(X_train, y_train)

# Model quality assessment
y_pred = model.predict(X_test)
print(f'Accuracy [test] : {accuracy_score(y_test, y_pred)}')

# save the model to disk
filename = 'model_perceptron.sav'
joblib.dump(model, filename)