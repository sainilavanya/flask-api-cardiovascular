import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data
        data = request.get_json()

        # Extract features (modify based on actual feature names)
        feature1 = data.get('feature1', 0)
        feature2 = data.get('feature2', 0)

        # Convert to numpy array for prediction (modify if more features exist)
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
    app.run(debug=True)
