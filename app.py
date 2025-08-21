from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    input_data = [
        float(request.form["app_usage"]),
        float(request.form["screen_time"]),
        float(request.form["battery_drain"]),
        float(request.form["apps_installed"]),
        float(request.form["data_usage"]),
        float(request.form["age"])
    ]
    scaled_data = scaler.transform([input_data])
    prediction = model.predict(scaled_data)[0]
    result = "Addicted" if prediction == 1 else "Not Addicted"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
