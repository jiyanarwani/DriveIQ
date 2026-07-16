from flask import Blueprint, jsonify, request
from backend.auth import token_required
from backend.db import get_dashboard_metrics

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/metrics", methods=["GET"])
@token_required
def get_metrics(current_user):
    try:
        metrics = get_dashboard_metrics(current_user["_id"])
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
