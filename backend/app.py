from flask import Flask, request, jsonify
from flask_cors import CORS
from services.predict_price_lite import predict_price_lite
from services.predict_rating_lite import predict_rating_lite
from services.predict_flavor import predict_flavor_tags_from_dict, load_model_eagerly, check_models_exist
from services.predict_mouthfeel import predict_mouthfeel_tags_from_dict
from mongo import init_mongo, db
import os

# Added imports for Google auth and JWT
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import jwt
from pymongo import ReturnDocument
from mongo import get_bottles_by_user
from mongo import add_bottle
from mongo import get_public_bottles
from mongo import now

app = Flask(__name__)
CORS(app)

# Initialize MongoDB
print("🗄️ Initializing MongoDB...")
MONGO_READY = init_mongo()

# JWT secret
JWT_SECRET = os.environ.get("JWT_SECRET", "change_me")

# Allowed Google audiences (comma-separated client IDs)
GOOGLE_CLIENT_IDS = [s.strip() for s in os.environ.get("GOOGLE_CLIENT_IDS", "").split(",") if s.strip()]

# Warm up the models on startup to avoid cold start penalty
print("🔥 Warming up models...")

# Warm up price and rating models first
try:
    warmup_data = {
        "variety": "Chardonnay",
        "country": "France", 
        "province": "Burgundy",
        "age": 3,
        "region_hierarchy": "France > Burgundy"
    }
    
    # Test price prediction
    price_warmup = predict_price_lite(warmup_data)
    print(f"✅ Price model warmed up successfully!")
    
    # Test rating prediction 
    rating_warmup_data = {
        **warmup_data,
        "price_min": float(price_warmup['weighted_lower']),
        "price_max": float(price_warmup['weighted_upper'])
    }
    rating_warmup = predict_rating_lite(rating_warmup_data)
    print(f"✅ Rating model warmed up successfully!")
    
except Exception as e:
    print(f"⚠️  Price/Rating model warmup failed: {e}")

# Warm up flavor prediction model - EAGERLY LOAD to avoid threading issues
try:
    print("🔍 Checking for flavor model files...")
    models_exist, missing_files = check_models_exist()
    
    if models_exist:
        print("✅ All flavor model files detected")
        print("🍷 Loading flavor model eagerly to avoid request-time delays...")
        
        # Load the model during startup to avoid threading issues
        success = load_model_eagerly()
        if success:
            print("✅ Flavor model loaded successfully during startup!")
            
            # Test a quick prediction to ensure everything works
            flavor_test_data = {
                **rating_warmup_data,
                "rating": float(rating_warmup['predicted_rating'])
            }
            test_flavors = predict_flavor_tags_from_dict(flavor_test_data, confidence_threshold=0.7, top_k=3)
            print(f"✅ Flavor model test successful! Found {len(test_flavors)} high-confidence flavors")
        else:
            print("⚠️  Flavor model loading failed during startup")
    else:
        print("⚠️  Some flavor model files not found:")
        for file in missing_files:
            print(f"    - {file}")
        print("    Note: Train the flavor model first with train_flavor_predictor.py")
        
except Exception as e:
    print(f"⚠️  Flavor model startup failed: {e}")
    print("    Note: Train the flavor model first with train_flavor_predictor.py")

# ---- Auth helpers and endpoints ----

def sign_app_jwt(payload: dict) -> str:
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_app_jwt(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])  # type: ignore


def require_auth(fn):
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
        if not token:
            return jsonify({"error": "Unauthorized"}), 401
        try:
            payload = decode_app_jwt(token)
            request.user = payload  # type: ignore
        except Exception:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


@app.route('/auth/google', methods=['POST'])
def auth_google():
    body = request.get_json(silent=True) or {}
    id_token_str = body.get('idToken')
    if not id_token_str:
        return jsonify({"error": "Missing idToken"}), 400
    try:
        ticket = google_id_token.verify_oauth2_token(id_token_str, google_requests.Request())
        if GOOGLE_CLIENT_IDS:
            aud = ticket.get('aud')
            if aud not in set(GOOGLE_CLIENT_IDS):
                return jsonify({"error": "Invalid audience"}), 401
        sub = ticket.get('sub')
        if not sub:
            return jsonify({"error": "Invalid token"}), 401
        email = ticket.get('email')
        name = ticket.get('name')
        picture = ticket.get('picture')

        user = db.users.find_one_and_update(
            {"google_id": sub},
            {
                "$setOnInsert": {"google_id": sub, "created_at": int(os.times().elapsed)},
                "$set": {"email": email, "name": name, "image": picture, "updated_at": int(os.times().elapsed)}
            },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )

        token = sign_app_jwt({"user_id": str(user.get("_id")), "google_id": sub})
        return jsonify({"token": token, "user": {"_id": str(user.get("_id")), "email": user.get("email"), "name": user.get("name"), "image": user.get("image")}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/me', methods=['GET'])
@require_auth
def me():
    user_id = request.user.get("user_id")  # type: ignore
    user = db.users.find_one({"_id": user_id})
    if not user:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"user": {"_id": str(user.get("_id")), "email": user.get("email"), "name": user.get("name"), "image": user.get("image")}})

@app.route('/cellar', methods=['GET'])
@require_auth
def get_cellar():
    user_google_id = request.user.get("google_id")  # type: ignore
    try:
        user = db.users.find_one({"google_id": user_google_id}, {"cellar": 1}) or {}
        items = user.get("cellar", [])
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cellar', methods=['POST'])
@require_auth
def create_bottle():
    user_id = request.user.get("user_id")  # type: ignore
    user_google_id = request.user.get("google_id")  # type: ignore
    body = request.get_json(silent=True) or {}
    try:
        # Normalize payload
        publish_public = bool(body.get("public", False))
        bottle_payload = {k: v for k, v in body.items() if k != "public"}
        bottle_payload.setdefault("created_at", now())

        public_id = None
        if publish_public:
            # Insert ONE public copy into global pool
            public_id = add_bottle(user_id, {**bottle_payload, "public": True})

        # Push an embedded copy into the user's document list (personal cellar)
        embedded_copy = {**bottle_payload, "saved_at": now(), "public": publish_public}
        if public_id is not None:
            embedded_copy["bottle_id"] = public_id
        db.users.update_one(
            {"google_id": user_google_id},
            {"$push": {"cellar": embedded_copy}},
            upsert=True,
        )

        return jsonify({"ok": True, "id": public_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cellar/public', methods=['GET'])
def list_public_cellar():
    try:
        items = get_public_bottles(limit=50)
        serialized = []
        for doc in items:
            doc_copy = dict(doc)
            if doc_copy.get("_id") is not None:
                doc_copy["_id"] = str(doc_copy["_id"])
            serialized.append(doc_copy)
        return jsonify({"items": serialized})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route('/predict-flavor', methods=['POST'])
def predict_flavor_endpoint():
    data = request.json
    print(f"Received data (flavor): {data}")
    
    try:
        # Extract parameters from request
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k = data.get('top_k', 10)
        
        # Check if we have all required basic fields
        required_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # If price_min/price_max not provided, predict them
        if 'price_min' not in data or 'price_max' not in data:
            print("  Predicting prices first...")
            price_prediction = predict_price_lite(data)
            data['price_min'] = float(price_prediction['weighted_lower'])
            data['price_max'] = float(price_prediction['weighted_upper'])
        
        # If rating not provided, predict it
        if 'rating' not in data:
            print("  Predicting rating first...")
            rating_prediction = predict_rating_lite(data)
            data['rating'] = float(rating_prediction['predicted_rating'])
        
        # Now predict flavors
        print("  Predicting flavors...")
        flavor_prediction = predict_flavor_tags_from_dict(
            data,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        return jsonify({
            "flavors": flavor_prediction,
            "input_data": {
                "variety": data['variety'],
                "country": data['country'],
                "province": data['province'],
                "age": data['age'],
                "region_hierarchy": data['region_hierarchy'],
                "price_min": data['price_min'],
                "price_max": data['price_max'],
                "rating": data['rating']
            },
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "top_k": top_k
            }
        })

    except Exception as e:
        print(f"Error (flavor): {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-mouthfeel', methods=['POST'])
def predict_mouthfeel_endpoint():
    data = request.json
    print(f"Received data (mouthfeel): {data}")
    
    try:
        # Extract parameters from request
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k = data.get('top_k', 10)
        
        # Check if we have all required basic fields
        required_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # If price_min/price_max not provided, predict them
        if 'price_min' not in data or 'price_max' not in data:
            print("  Predicting prices first...")
            price_prediction = predict_price_lite(data)
            data['price_min'] = float(price_prediction['weighted_lower'])
            data['price_max'] = float(price_prediction['weighted_upper'])
        
        # If rating not provided, predict it
        if 'rating' not in data:
            print("  Predicting rating first...")
            rating_prediction = predict_rating_lite(data)
            data['rating'] = float(rating_prediction['predicted_rating'])
        
        # Now predict mouthfeel
        print("  Predicting mouthfeel...")
        mouthfeel_prediction = predict_mouthfeel_tags_from_dict(
            data,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        result = {
            "mouthfeel": mouthfeel_prediction,
            "input_data": data,
            "prediction_info": {
                "confidence_threshold": confidence_threshold,
                "top_k": top_k,
                "total_found": len(mouthfeel_prediction)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in mouthfeel prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict-all', methods=['POST'])
def predict_all_endpoint():
    """Convenience endpoint that predicts price, rating, flavors, and mouthfeel all at once"""
    data = request.json
    print(f"Received data (all): {data}")
    
    try:
        # Extract prediction parameters
        confidence_threshold = data.get('confidence_threshold', 0.5)
        top_k = data.get('top_k', 10)
        
        # Check required fields
        required_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Step 1: Predict price
        print("  Step 1: Predicting price...")
        price_prediction = predict_price_lite(data)
        price_min = float(price_prediction['weighted_lower'])
        price_max = float(price_prediction['weighted_upper'])
        
        # Step 2: Predict rating (using price predictions)
        print("  Step 2: Predicting rating...")
        rating_input = {
            **data,
            "price_min": price_min,
            "price_max": price_max
        }
        rating_prediction = predict_rating_lite(rating_input)
        rating = float(rating_prediction['predicted_rating'])
        
        # Step 3: Predict flavors (using price and rating predictions)
        print("  Step 3: Predicting flavors...")
        prediction_input = {
            **data,
            "price_min": price_min,
            "price_max": price_max,
            "rating": rating
        }
        flavor_prediction = predict_flavor_tags_from_dict(
            prediction_input,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        # Step 4: Predict mouthfeel (using price and rating predictions)
        print("  Step 4: Predicting mouthfeel...")
        mouthfeel_prediction = predict_mouthfeel_tags_from_dict(
            prediction_input,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        # Combine all results
        result = {
            "price": price_prediction,
            "rating": rating_prediction,
            "flavors": flavor_prediction,
            "mouthfeel": mouthfeel_prediction,
            "input_data": data,
            "prediction_info": {
                "confidence_threshold": confidence_threshold,
                "top_k": top_k,
                "flavor_count": len(flavor_prediction),
                "mouthfeel_count": len(mouthfeel_prediction)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in combined prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    mongo_status = "connected"
    try:
        db.command('ping')
    except Exception as e:
        mongo_status = f"error: {e.__class__.__name__}"

    return jsonify({
        "status": "healthy",
        "models": {
            "price": "available",
            "rating": "available", 
            "flavor": "available" if 'predict_flavor_tags_from_dict' in globals() else "training_required"
        },
        "mongo": {
            "status": mongo_status,
            "ready_at_startup": bool(MONGO_READY)
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 Available endpoints:")
    print("   POST /auth/google        - Verify Google ID token and return app JWT")
    print("   GET  /me                 - Get current user by JWT")
    print("   POST /predict-price-lite   - Predict wine price range")
    print("   POST /predict-rating-lite  - Predict wine rating")
    print("   POST /predict-flavor       - Predict wine flavors")
    print("   POST /predict-all          - Predict price, rating, and flavors")
    print("   GET  /health               - Health check")
    app.run(debug=True, host='0.0.0.0', port=5001)
