from flask import Flask, render_template, request, jsonify
import flask
import joblib
import numpy as np
from pip import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def WeightPredict():
    if request.method == 'GET':
        return render_template("WeightPredictDashboard.html")
    elif request.method == 'POST':
        print(dict(request.form))
        WeightPredict = dict(request.form).values()
        WeightPredict = np.array([float(x) for x in WeightPredict])
        model, std_scaler = joblib.load("Model/model.pkl")
        WeightPredict = std_scaler.transform([WeightPredict])
        print(WeightPredict)
        result = model.predict(WeightPredict)
        result = "%.2f" % result
        return render_template('WeightPredictDashboard.html', result=result)
    else:
        return "Ada Kesalahan Nich, Debugging lagi yuk..."

if __name__ == '__main__':
    app.run(port=5000, debug=True)