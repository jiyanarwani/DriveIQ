from fastapi.testclient import TestClient
from backend.app import app
from backend.db import sessions_collection
from backend.auth import generate_token
import uuid

def test():
    token = generate_token("test_user_123")
    client = TestClient(app)
    
    session_id = f"test-trip-{uuid.uuid4()}"
    payload = {
        "session_id": session_id,
        "scoring_mode": "xgboost",
        "telemetry": {
            "speed": 80.0,
            "acceleration": -2.0,  # hard braking
            "throttle_position": 0.0
        }
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    resp = client.post("/api/v1/score", json=payload, headers=headers)
    print("Response status:", resp.status_code)
    print("Response body:", resp.json())
    
    # Check DB if connected
    if sessions_collection is not None:
        try:
            doc = sessions_collection.find_one({"session_id": session_id})
            if doc:
                print("SUCCESS! Document found in DB with session_id:", doc["session_id"])
                print("Events saved:", doc.get("events"))
        except Exception as e:
            print("DB check skipped (MongoDB not running):", e)

if __name__ == "__main__":
    test()
