import requests

API_URL = "http://127.0.0.1:8000/api/predict"

def predict_image(file):
    files = {"file": file}
    response = requests.post(API_URL, files=files)

    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code != 200:
        return {"error": response.text}

    return response.json()