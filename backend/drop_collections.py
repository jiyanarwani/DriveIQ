import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["DriveIQ"]
db.sessions.drop()
db.trip_summaries.drop()
print("Dropped old collections to apply new schema!")
