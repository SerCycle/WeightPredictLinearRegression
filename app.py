from flask import Flask, render_template, request, jsonify
import flask
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def WeightPredict():
    
    # while First Landing Page
    if request.method == 'GET':
        return render_template("WeightPredictDashboard.html")
    
    # After give value
    elif request.method == 'POST':
        # mengambil value POST method
        gender = request.form['gender']
        if gender == "1":
            gender = "Laki-laki"
        elif gender == "2":
            gender = "Perempuan"
        else:
            gender ="???"
        height = request.form['tinggi']
        print(dict(request.form))
        WeightPredict = dict(request.form).values()
        WeightPredict = np.array([float(x) for x in WeightPredict])
        # Load Model
        model, std_scaler = joblib.load("Model/model.pkl")
        WeightPredict = std_scaler.transform([WeightPredict])
        print(WeightPredict)
        result = model.predict(WeightPredict)
        result = "%.2f" % result
        return render_template('WeightPredictDashboard.html', result=result, gender=gender, tinggi=height)
    
    # If Something Wrong
    else:
        return "Ada Kesalahan Nich, Debugging lagi yuk..."

if __name__ == '__main__':
    app.run(port=5000, debug=True)