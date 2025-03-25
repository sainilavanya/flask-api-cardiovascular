from flask import Flask, request, jsonify
import joblib  # For loading the model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")  # Make sure model.pkl is in /content/

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
