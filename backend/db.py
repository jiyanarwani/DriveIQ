from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from pymongo.errors import PyMongoError

load_dotenv()
logger = logging.getLogger("driveiq.db")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing required env var: MONGO_URI")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["DriveIQ"]

users_collection = db["users"]
sessions_collection = db["sessions"]
trip_summaries_collection = db["trip_summaries"]

# Keep auth deterministic and avoid duplicate email races.
try:
    users_collection.create_index("email", unique=True)
except PyMongoError as e:
    # Do not crash the API process at import time if Mongo is temporarily down.
    logger.warning("Mongo index init skipped: %s", e)


def is_mongo_available() -> tuple[bool, str | None]:
    try:
        client.admin.command("ping")
        return True, None
    except Exception as e:
        return False, str(e)

def save_session(user_id, session_id, score, eco_score, features, events):
    now = datetime.utcnow()
    sessions_collection.update_one(
        {"user_id": user_id, "session_id": session_id},
        {
            "$setOnInsert": {
                "start_time": now,
                "tips": [],
                "aggregated_features": {}
            },
            "$set": {
                "end_time": now,
                "final_score": score
            },
            "$push": {
                "frames": {
                    "timestamp": now,
                    "score": score,
                    "eco_score": eco_score,
                    "features": features,
                    "events": events
                }
            },
            "$inc": {
                "frame_count": 1
            }
        },
        upsert=True
    )

def get_user_sessions(user_id, limit=50):
    return list(sessions_collection.find({"user_id": user_id}).sort("start_time", -1).limit(limit))

def get_dashboard_metrics(user_id):
    sessions = list(sessions_collection.find({"user_id": user_id}))
    if not sessions:
        return {
            "mean_eco_score": 0,
            "lowest_eco_score": 0,
            "total_trips": 0,
            "trips_this_week": 0
        }
    
    total_trips = len(sessions)
    scores = [s.get("final_score", 0) for s in sessions]
    mean_eco_score = sum(scores) / len(scores) if scores else 0
    lowest_eco_score = min(scores) if scores else 0
    
    now = datetime.utcnow()
    trips_this_week = 0
    for s in sessions:
        ts = s.get("start_time")
        if ts and (now - ts).days <= 7:
            trips_this_week += 1
            
    return {
        "mean_eco_score": round(mean_eco_score),
        "lowest_eco_score": round(lowest_eco_score),
        "total_trips": total_trips,
        "trips_this_week": trips_this_week
    }
