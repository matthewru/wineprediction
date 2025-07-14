import requests

payload = {
    "variety": "Merlot",
    "country": "US",
    "province": "California",
    "age": 3,
    "region_hierarchy": "US > California > Sonoma County"
}

r_price = requests.post("http://localhost:5001/predict-price-lite", json=payload)
r_rating = requests.post("http://localhost:5001/predict-rating-lite", json=payload)

print(r_price.status_code)
print(r_price.text)
print(r_rating.status_code)
print(r_rating.text)
