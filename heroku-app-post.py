import requests
import json

url = "https://census-classification-app-beee7d84c1fe.herokuapp.com/predict"
body = {"path": "data/test-data.csv"}

r = requests.post(url, data=json.dumps(body))
print("status code is", r.status_code)
print("predictions are", r.json())
