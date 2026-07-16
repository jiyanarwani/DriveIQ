import json
from backend.app import app
from backend.db import sessions_collection
from backend.auth import generate_token
import uuid

def test():
    with app.app_context():
        # Clear previous test data
        sessions_collection.delete_many({"user_id": "test_user_123"})
        
        token = generate_token("test_user_123")
        client = app.test_client()
        
        session_id = f"test-trip-{uuid.uuid4()}"
        payload = {
            "session_id": session_id,
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
        
        resp = client.post("/api/score", data=json.dumps(payload), headers=headers)
        print("Response status:", resp.status_code)
        print("Response body:", resp.json)
        
        # Check DB
        doc = sessions_collection.find_one({"session_id": session_id})
        if doc:
            print("SUCCESS! Document found in DB with session_id:", doc["session_id"])
            print("Events saved:", doc.get("events"))
        else:
            print("ERROR: Document not found in DB.")

if __name__ == "__main__":
    test()
