from flask import Blueprint, request, jsonify
import logging
from backend.db import users_collection, is_mongo_available
from backend.auth import generate_token
import bcrypt
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

logger = logging.getLogger("driveiq.routes.auth")

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    mongo_ok, mongo_error = is_mongo_available()
    if not mongo_ok:
        return jsonify({
            "error": "Database unavailable",
            "message": "MongoDB is not reachable. Start MongoDB and retry.",
            "details": mongo_error,
        }), 503

    try:
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "User exists"}), 400

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        user_id = users_collection.insert_one({"email": email, "password": hashed}).inserted_id

        token = generate_token(str(user_id))
        return jsonify({"token": token}), 201
    except DuplicateKeyError:
        return jsonify({"error": "User exists"}), 400
    except ServerSelectionTimeoutError as e:
        return jsonify({
            "error": "Database unavailable",
            "message": "MongoDB connection timed out.",
            "details": str(e),
        }), 503
    except Exception as e:
        logger.exception("Registration failed: %s", e)
        return jsonify({"error": "Registration service unavailable"}), 503

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    mongo_ok, mongo_error = is_mongo_available()
    if not mongo_ok:
        return jsonify({
            "error": "Database unavailable",
            "message": "MongoDB is not reachable. Start MongoDB and retry.",
            "details": mongo_error,
        }), 503

    try:
        user = users_collection.find_one({"email": email})
        if not user or not bcrypt.checkpw(password.encode(), user["password"]):
            return jsonify({"error": "Invalid credentials"}), 401

        token = generate_token(str(user["_id"]))
        return jsonify({"token": token}), 200
    except ServerSelectionTimeoutError as e:
        return jsonify({
            "error": "Database unavailable",
            "message": "MongoDB connection timed out.",
            "details": str(e),
        }), 503
    except Exception as e:
        logger.exception("Login failed: %s", e)
        return jsonify({"error": "Login service unavailable"}), 503
