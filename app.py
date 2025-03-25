from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Cardiovascular Disease Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return jsonify({"message": "Prediction route is working", "received_data": data})

if __name__ == '__main__':
    from os import environ
    port = int(environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host='0.0.0.0', port=port)
