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
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")

    # Test predict-price endpoint
    print("1. Predict-Price:")
    try:
        r_price = requests.post("http://localhost:5001/predict-price-lite", json=payload_data, timeout=10)
        print(f"   Status: {r_price.status_code}")
        if r_price.status_code == 200:
            price_data = r_price.json()
            print(f"   Price Range: {price_data.get('weighted_range', 'N/A')}")
        else:
            print(f"   Error: {r_price.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

    # Test predict-rating endpoint
    print("2. Predict-Rating:")
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

    # Test predict-flavor endpoint
    print("3. Predict-Flavor:")
    flavor_payload = {
        **payload_data,
        "confidence_threshold": 0.2,  # Changed from 0.3 to match predict-all
        "top_k": 8
    }
    
    try:
        r_flavor = requests.post("http://localhost:5001/predict-flavor", json=flavor_payload, timeout=10)
        print(f"   Status: {r_flavor.status_code}")
        if r_flavor.status_code == 200:
            flavor_data = r_flavor.json()
            flavors = flavor_data.get('flavors', [])
            print(f"   Found {len(flavors)} flavor tags:")
            for i, flavor in enumerate(flavors[:8]):  # Show top 8
                print(f"     {i+1}. {flavor['flavor']:<15} ({flavor['confidence']:.3f})")
            if len(flavors) > 8:
                print(f"     ... and {len(flavors) - 8} more")
        else:
            print(f"   Error: {r_flavor.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

    # Test predict-mouthfeel endpoint
    print("4. Predict-Mouthfeel:")
    mouthfeel_payload = {
        **payload_data,
        "confidence_threshold": 0.2,  # Changed from 0.3 to match predict-all
        "top_k": 8
    }
    
    try:
        r_mouthfeel = requests.post("http://localhost:5001/predict-mouthfeel", json=mouthfeel_payload, timeout=10)
        print(f"   Status: {r_mouthfeel.status_code}")
        if r_mouthfeel.status_code == 200:
            mouthfeel_data = r_mouthfeel.json()
            mouthfeel = mouthfeel_data.get('mouthfeel', [])
            print(f"   Found {len(mouthfeel)} mouthfeel tags:")
            for i, feel in enumerate(mouthfeel[:8]):  # Show top 8
                print(f"     {i+1}. {feel['mouthfeel']:<15} ({feel['confidence']:.3f})")
            if len(mouthfeel) > 8:
                print(f"     ... and {len(mouthfeel) - 8} more")
        else:
            print(f"   Error: {r_mouthfeel.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error: {e}")

    # Test predict-all endpoint
    print("5. Predict-All:")
    all_payload = {
        **payload_data,
        "confidence_threshold": 0.2,  # Lowered from 0.4 to get more tags  
        "top_k": 10  # Increased from 5 to have more room
    }
    
    try:
        r_all = requests.post("http://localhost:5001/predict-all", json=all_payload, timeout=15)
        print(f"   Status: {r_all.status_code}")
        if r_all.status_code == 200:
            all_data = r_all.json()
            
            # Price
            price_info = all_data.get('price', {})
            print(f"   Price: {price_info.get('weighted_range', 'N/A')}")
            
            # Rating
            rating_info = all_data.get('rating', {})
            print(f"   Rating: {rating_info.get('predicted_rating', 'N/A'):.1f}")
            
            # Flavors
            flavors = all_data.get('flavors', [])
            print(f"   Flavors ({len(flavors)}): {', '.join([f['flavor'] for f in flavors[:5]])}")
            
            # Mouthfeel
            mouthfeel = all_data.get('mouthfeel', [])
            print(f"   Mouthfeel ({len(mouthfeel)}): {', '.join([m['mouthfeel'] for m in mouthfeel[:5]])}")
            
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
