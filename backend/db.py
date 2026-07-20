from pymongo import MongoClient
from datetime import datetime
import logging
from pymongo.errors import PyMongoError
from backend.config import settings

logger = logging.getLogger("driveiq.db")

# Try to connect to MongoDB, fall back to mock database if unavailable
mongo_available = False
client = None
db = None

try:
    client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=2000)
    # Ping database to verify connection
    client.admin.command("ping")
    db = client["DriveIQ"]
    mongo_available = True
    logger.info("✅ MongoDB connected successfully.")
except Exception as e:
    logger.warning(f"⚠️ MongoDB connection failed ({e}). Falling back to in-memory Mock Database for development.")

# ── Mock Database Implementation ──────────────────────────────────────────────

class MockCollection:
    def __init__(self, name: str):
        self.name = name
        self.data: dict[str, dict] = {}

    def find_one(self, filter: dict) -> dict | None:
        for doc in self.data.values():
            match = True
            for k, v in filter.items():
                # Allow matching stringified ObjectIds or normal values
                doc_val = doc.get(k)
                if str(doc_val) != str(v):
                    match = False
                    break
            if match:
                # Return copy so internal state isn't accidentally modified
                return dict(doc)
        return None

    def insert_one(self, doc: dict):
        from bson import ObjectId
        new_doc = dict(doc)
        if "_id" not in new_doc:
            new_doc["_id"] = ObjectId()
        self.data[str(new_doc["_id"])] = new_doc
        
        class Result:
            inserted_id = new_doc["_id"]
        return Result()

    def update_one(self, filter: dict, update: dict, upsert: bool = False):
        target = None
        for doc in self.data.values():
            match = True
            for k, v in filter.items():
                if str(doc.get(k)) != str(v):
                    match = False
                    break
            if match:
                target = doc
                break

        if not target:
            if upsert:
                target = {k: v for k, v in filter.items()}
                if "$setOnInsert" in update:
                    for k, v in update["$setOnInsert"].items():
                        target[k] = v
                res = self.insert_one(target)
                target = self.data[str(res.inserted_id)]
            else:
                return

        if "$set" in update:
            for k, v in update["$set"].items():
                target[k] = v
        if "$push" in update:
            for k, v in update["$push"].items():
                if k not in target:
                    target[k] = []
                target[k].append(v)
        if "$inc" in update:
            for k, v in update["$inc"].items():
                target[k] = target.get(k, 0) + v

        self.data[str(target["_id"])] = target

    def find(self, filter: dict | None = None):
        results = []
        for doc in self.data.values():
            if not filter:
                results.append(dict(doc))
                continue
            match = True
            for k, v in filter.items():
                if str(doc.get(k)) != str(v):
                    match = False
                    break
            if match:
                results.append(dict(doc))

        class MockCursor:
            def __init__(self, data):
                self.data = data

            def sort(self, key, direction=-1):
                if key == "start_time":
                    self.data = sorted(
                        self.data, 
                        key=lambda x: x.get("start_time", datetime.min), 
                        reverse=(direction == -1)
                    )
                return self

            def limit(self, n):
                self.data = self.data[:n]
                return self

            def __iter__(self):
                return iter(self.data)

        return MockCursor(results)

    def create_index(self, keys, **kwargs):
        pass


# Initialize collections
if mongo_available and db is not None:
    users_collection = db["users"]
    sessions_collection = db["sessions"]
    trip_summaries_collection = db["trip_summaries"]
    
    try:
        users_collection.create_index("email", unique=True)
    except PyMongoError as e:
        logger.warning("Mongo index init skipped: %s", e)
else:
    users_collection = MockCollection("users")
    sessions_collection = MockCollection("sessions")
    trip_summaries_collection = MockCollection("trip_summaries")


def is_mongo_available() -> tuple[bool, str | None]:
    if not mongo_available:
        return True, "in_memory_mock_fallback_active"
    try:
        if client is not None:
            client.admin.command("ping")
            return True, None
        return False, "MongoDB client not initialized"
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
