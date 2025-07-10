from flask import Flask, request, jsonify
from flask_cors import CORS
from services.predict_price_lite import predict_price_lite

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict-price-lite', methods=['POST'])
def predict_price_lite_endpoint():
    data = request.json
    print(f"Received data: {data}")  # Debug print
    try:
        prediction = predict_price_lite(data)
        return jsonify(prediction)
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)


