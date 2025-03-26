from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Cardiovascular Disease Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        model = joblib.load("model.pkl")  # Ensure model.pkl exists
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    from os import environ
    port = int(environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host='0.0.0.0', port=port)
