import numpy as np 
from flask import Flask, jsonify, request, render_template
import joblib

#create flas app
app = Flask(__name__)

#load the pickle model
model = joblib.load("iris_KNN_model.pkl")

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"]) # POST because we are going to receive the inputs : independent variables
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main_":
    app.run(host = '0.0.0.0', port = '4000', debug=True)
