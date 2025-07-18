import requests
import json

# Test payloads - various wine types and regions
test_payloads = [
    {
        "name": "Austrian Pinot Grigio",
        "payload": {
            "variety": "Pinot Grigio",
            "country": "Austria",
            "province": "Unknown",
            "age": 3,
            "region_hierarchy": "Austria > Unknown > Unknown"
        }
    },
    {
        "name": "French Chardonnay",
        "payload": {
            "variety": "Chardonnay",
            "country": "France",
            "province": "Burgundy",
            "age": 5,
            "region_hierarchy": "France > Burgundy > Chablis"
        }
    },
    {
        "name": "California Cabernet Sauvignon",
        "payload": {
            "variety": "Cabernet Sauvignon",
            "country": "US",
            "province": "California",
            "age": 7,
            "region_hierarchy": "US > California > Napa Valley"
        }
    },
    {
        "name": "Italian Chianti",
        "payload": {
            "variety": "Sangiovese",
            "country": "Italy",
            "province": "Tuscany",
            "age": 4,
            "region_hierarchy": "Italy > Tuscany > Chianti"
        }
    },
    {
        "name": "Spanish Tempranillo",
        "payload": {
            "variety": "Tempranillo",
            "country": "Spain",
            "province": "Rioja",
            "age": 6,
            "region_hierarchy": "Spain > Rioja > Rioja Alta"
        }
    },
    {
        "name": "German Riesling",
        "payload": {
            "variety": "Riesling",
            "country": "Germany",
            "province": "Mosel",
            "age": 2,
            "region_hierarchy": "Germany > Mosel > Bernkastel"
        }
    },
    {
        "name": "Australian Shiraz",
        "payload": {
            "variety": "Shiraz",
            "country": "Australia",
            "province": "South Australia",
            "age": 8,
            "region_hierarchy": "Australia > South Australia > Barossa Valley"
        }
    },
    {
        "name": "New Zealand Sauvignon Blanc",
        "payload": {
            "variety": "Sauvignon Blanc",
            "country": "New Zealand",
            "province": "Marlborough",
            "age": 1,
            "region_hierarchy": "New Zealand > Marlborough > Marlborough"
        }
    },
    {
        "name": "Oregon Pinot Noir",
        "payload": {
            "variety": "Pinot Noir",
            "country": "US",
            "province": "Oregon",
            "age": 3,
            "region_hierarchy": "US > Oregon > Willamette Valley"
        }
    },
    {
        "name": "Portuguese Red Blend",
        "payload": {
            "variety": "Portuguese Red",
            "country": "Portugal",
            "province": "Douro",
            "age": 10,
            "region_hierarchy": "Portugal > Douro > Porto"
        }
    }
]

def test_wine_prediction(payload_data, test_name):
    """Test a single wine payload against all endpoints"""
    print(f"\n🍷 Testing: {test_name}")
    print("=" * 60)
    print(f"Wine details: {payload_data}")
    print("-" * 60)
    
    # Test price prediction
    print("1. Price Prediction:")
    try:
        r_price = requests.post("http://localhost:5001/predict-price-lite", json=payload_data, timeout=10)
        print(f"   Status: {r_price.status_code}")
        if r_price.status_code == 200:
            price_data = r_price.json()
            print(f"   Price Range: {price_data.get('weighted_range', 'N/A')}")
            print(f"   Top Bucket: {price_data.get('top_bucket', 'N/A')}")
        else:
            print(f"   Error: {r_price.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

    # Test rating prediction
    print("2. Rating Prediction:")
    try:
        r_rating = requests.post("http://localhost:5001/predict-rating-lite", json=payload_data, timeout=10)
        print(f"   Status: {r_rating.status_code}")
        if r_rating.status_code == 200:
            rating_data = r_rating.json()
            print(f"   Predicted Rating: {rating_data.get('predicted_rating', 'N/A'):.1f}")
        else:
            print(f"   Error: {r_rating.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

    # Test flavor prediction
    print("3. Flavor Prediction:")
    flavor_payload = {
        **payload_data,
        "confidence_threshold": 0.3,
        "top_k": 10
    }
    try:
        r_flavor = requests.post("http://localhost:5001/predict-flavor", json=flavor_payload, timeout=10)
        print(f"   Status: {r_flavor.status_code}")
        if r_flavor.status_code == 200:
            flavor_data = r_flavor.json()
            flavors = flavor_data.get('flavors', [])
            print(f"   Found {len(flavors)} flavor tags:")
            for i, flavor in enumerate(flavors[:5]):  # Show top 5
                print(f"     {i+1}. {flavor['flavor']:<15} ({flavor['confidence']:.3f})")
            if len(flavors) > 5:
                print(f"     ... and {len(flavors) - 5} more")
        else:
            print(f"   Error: {r_flavor.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

    # Test predict-all endpoint
    print("4. Predict-All:")
    all_payload = {
        **payload_data,
        "confidence_threshold": 0.4,
        "top_k": 5
    }
    try:
        r_all = requests.post("http://localhost:5001/predict-all", json=all_payload, timeout=10)
        print(f"   Status: {r_all.status_code}")
        if r_all.status_code == 200:
            all_data = r_all.json()
            
            # Price info
            price_info = all_data.get('price', {})
            print(f"   Price: {price_info.get('weighted_range', 'N/A')}")
            
            # Rating info
            rating_info = all_data.get('rating', {})
            print(f"   Rating: {rating_info.get('predicted_rating', 'N/A'):.1f}")
            
            # Top flavors
            flavors = all_data.get('flavors', [])
            if flavors:
                print(f"   Top flavors: {', '.join([f['flavor'] for f in flavors[:3]])}")
            
            # Derived features
            derived = all_data.get('derived_features', {})
            if derived:
                print(f"   Price range: ${derived.get('price_min', 0):.2f} - ${derived.get('price_max', 0):.2f}")
        else:
            print(f"   Error: {r_all.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

print("🍷 Wine Prediction API - Comprehensive Testing")
print("=" * 70)

# Test health check first
print("\n🏥 Health Check:")
try:
    r_health = requests.get("http://localhost:5001/health", timeout=5)
    print(f"Status: {r_health.status_code}")
    if r_health.status_code == 200:
        health_data = r_health.json()
        print(f"Overall Status: {health_data.get('status', 'unknown')}")
        models = health_data.get('models', {})
        for model, status in models.items():
            print(f"  {model.capitalize()} model: {status}")
    else:
        print(f"Error: {r_health.text}")
        print("⚠️  Server may not be running or models not loaded properly")
except requests.exceptions.RequestException as e:
    print(f"Connection Error: {e}")
    print("❌ Cannot connect to server. Make sure it's running on http://localhost:5001")

# Run tests for each payload
for i, test_case in enumerate(test_payloads, 1):
    test_wine_prediction(test_case["payload"], f"{i}/10 - {test_case['name']}")
    
    # Add a small separator between tests
    if i < len(test_payloads):
        print("\n" + "." * 40 + "\n")

print("\n" + "=" * 70)
print("🎉 Comprehensive API testing complete!")
print(f"📊 Tested {len(test_payloads)} different wine varieties from around the world")
