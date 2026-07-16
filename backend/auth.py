import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from functools import wraps
from flask import request, jsonify


load_dotenv()

SECRET = os.getenv("JWT_SECRET")
if not SECRET:
    raise RuntimeError("Missing required env var: JWT_SECRET")
if len(SECRET) < 32:
    raise RuntimeError("JWT_SECRET must be at least 32 characters long")

def generate_token(user_id):
    return jwt.encode({
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }, SECRET, algorithm="HS256")

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"error": "Token is missing"}), 401

        user_id = verify_token(token)
        if not user_id:
            return jsonify({"error": "Token is invalid"}), 401

        current_user = {"_id": user_id}
        return f(current_user, *args, **kwargs)

    return decorated
