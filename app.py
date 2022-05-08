from flask import Flask
from flask_restful import Api
from flask import request, render_template
import joblib
import numpy as np


app = Flask(__name__)
model = joblib.load('model_perceptron.sav')

@app.route('/')
def form():    
    return render_template('form.html')

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == 'POST':
        form_data = request.form
        form_data_dict = form_data.to_dict()
        X = np.array([float(val) for val in form_data_dict.values()])
        y_pred = model.predict_obs(X)
        return render_template('data.html', form_data=form_data, y_pred=y_pred)

if __name__ == '__main__':
    app.run(port=5010)
