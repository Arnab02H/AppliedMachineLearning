import os 
import time 
import requests
import subprocess 
import signal 

def test_flask():
    process=subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(3)

    try:
        url="http://127.0.0.1:5000/score"
        payload = {
            "text": "Congratulations! You won 1000 dollars!",
            "threshold": 0.5
        }
        response=requests.post(url,json=payload)

        assert response.status_code==200 
        data=response.json()

        assert "prediction" in data
        assert "propensity" in data

        assert isinstance(data["prediction"], bool)
        assert 0.0 <= data["propensity"] <= 1.0

    finally:
        process.terminate()
        process.wait()