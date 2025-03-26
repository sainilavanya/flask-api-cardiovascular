import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)  
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get the input JSON data
        data = request.get_json()

        # Extract features safely
        feature1 = float(data.get('feature1', 0))
        feature2 = float(data.get('feature2', 0))

        # Convert to numpy array for prediction
        input_data = np.array([[feature1, feature2]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return prediction as JSON
        return jsonify({
            "message": "Prediction successful",
            "prediction": prediction.tolist()  # Convert NumPy array to list
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Fixed comment syntax
