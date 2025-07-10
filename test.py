import requests

payload = {
    "variety": "Merlot",
    "country": "US",
    "province": "California",
    "age": 3,
    "region_hierarchy": "US > California > Sonoma County"
}

r = requests.post("http://localhost:5001/predict-price-lite", json=payload)

print(r.status_code)
print(r.text)
