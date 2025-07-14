from flask import Flask, request, jsonify
from flask_cors import CORS
from services.predict_price_lite import predict_price_lite
from services.predict_rating_lite import predict_rating_lite  # Assumes this function exists and works

app = Flask(__name__)
CORS(app)

@app.route('/predict-price-lite', methods=['POST'])
def predict_price_lite_endpoint():
    data = request.json
    print(f"Received data (price): {data}")
    try:
        prediction = predict_price_lite(data)
        return jsonify(prediction)
    except Exception as e:
        print(f"Error (price): {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-rating-lite', methods=['POST'])
def predict_rating_lite_endpoint():
    data = request.json
    print(f"Received data (rating): {data}")

    try:
        # Step 1: Predict price bounds from user input
        price_prediction = predict_price_lite(data)
        price_min = float(price_prediction['weighted_lower'])
        price_max = float(price_prediction['weighted_upper'])

        # Step 2: Add these to input and call rating model
        full_input = {
            **data,
            "price_min": price_min,
            "price_max": price_max
        }

        # Step 3: Predict rating
        rating_prediction = predict_rating_lite(full_input)
        return jsonify(rating_prediction)

    except Exception as e:
        print(f"Error (rating): {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
