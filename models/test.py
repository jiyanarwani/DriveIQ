"""Basic API contract smoke tests for DriveIQ backend.

Run:
	source .venv/bin/activate
	python test.py
"""

import base64
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backend.app import create_app


class DriveIQApiTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		# Force fallback coaching path for deterministic offline testing.
		os.environ.pop("ANTHROPIC_API_KEY", None)
		os.environ["DRIVEIQ_DISABLE_FLAN_COACH"] = "1"

	def setUp(self):
		app = create_app()
		app.testing = True
		self.app = app
		self.client = app.test_client()

	def _tiny_jpg_b64(self, width=64, height=48):
		"""Create a deterministic tiny JPEG payload for frame-route tests."""
		img = np.zeros((height, width, 3), dtype=np.uint8)
		cv2.rectangle(img, (8, 8), (width - 8, height - 8), (255, 255, 255), 2)
		ok, enc = cv2.imencode(".jpg", img)
		self.assertTrue(ok)
		return base64.b64encode(enc.tobytes()).decode("utf-8")

	def test_health_contract(self):
		r = self.client.get("/api/health")
		self.assertEqual(r.status_code, 200)
		data = r.get_json()

		self.assertEqual(data.get("status"), "ok")
		self.assertIn("models_loaded", data)
		self.assertIn("core_models_loaded", data)
		self.assertIn("schema_valid", data)
		self.assertIn("schema_error", data)
		self.assertIn("version", data)
		self.assertEqual(data["models_loaded"], data["core_models_loaded"])
		self.assertIsInstance(data["missing_core_models"], list)

		# Current readiness: xgb+scaler+schema.
		models = self.app.config["MODELS"]
		expected_core = bool(
			models.get("xgb") is not None
			and models.get("scaler") is not None
			and models.get("schema_valid")
		)
		self.assertEqual(data["core_models_loaded"], expected_core)

	def test_health_schema_invalid_marks_not_ready(self):
		models = self.app.config["MODELS"]
		old_schema_valid = models.get("schema_valid")
		old_schema_error = models.get("schema_error")
		try:
			models["schema_valid"] = False
			models["schema_error"] = "forced-health-schema-test"
			r = self.client.get("/api/health")
			self.assertEqual(r.status_code, 200)
			data = r.get_json()
			self.assertFalse(data["schema_valid"])
			self.assertFalse(data["models_loaded"])
			self.assertFalse(data["core_models_loaded"])
		finally:
			models["schema_valid"] = old_schema_valid
			models["schema_error"] = old_schema_error

	def test_score_contract(self):
		payload = {
			"session_id": "test-session",
			"telemetry": {
				"speed": 62,
				"rpm": 2200,
				"throttle_position": 28,
				"gear": 4,
				"acceleration": 0.4,
				"fuel_rate": 7.3,
			},
		}
		r = self.client.post("/api/score", json=payload)
		self.assertEqual(r.status_code, 200)
		data = r.get_json()

		self.assertIn("score", data)
		self.assertIn("features", data)
		self.assertIsInstance(data["score"], (int, float))
		self.assertTrue(0 <= data["score"] <= 100)

	def test_score_invalid_frame_has_cv_error(self):
		payload = {
			"session_id": "test-session-invalid-frame",
			"telemetry": {
				"speed": 55,
				"rpm": 1800,
				"throttle_position": 22,
				"gear": 3,
				"acceleration": -0.2,
			},
			"frame_b64": "not-base64",
		}
		r = self.client.post("/api/score", json=payload)
		self.assertEqual(r.status_code, 200)
		data = r.get_json()
		self.assertIn("features", data)
		self.assertIn("cv_error", data["features"])

	def test_score_with_frame_chain_contract(self):
		frame1 = self._tiny_jpg_b64()
		frame2 = self._tiny_jpg_b64()
		payload = {
			"session_id": "test-session-frame-chain",
			"telemetry": {
				"speed": 42,
				"rpm": 1600,
				"throttle_position": 18,
				"gear": 3,
				"acceleration": 0.1,
			},
			"frame_b64": frame2,
			"prev_frame_b64": frame1,
		}
		mock_feats = {
			"vehicle_count": 2,
			"proximity_score": 0.05,
			"pedestrian_flag": 0,
			"mean_flow": 1.2,
			"flow_variance": 0.7,
			"braking_flag": 0,
			"lane_change_flag": 0,
			"road_type_id": 0,
			"weather_id": 0,
		}
		with patch("cv.cv_pipeline.cv_pipeline", return_value=mock_feats):
			r = self.client.post("/api/score", json=payload)
		self.assertEqual(r.status_code, 200)
		data = r.get_json()
		self.assertIn("score", data)
		self.assertIn("features", data)
		self.assertTrue(0 <= data["score"] <= 100)

	def test_score_schema_mismatch_failfast(self):
		models = self.app.config["MODELS"]
		old_valid = models.get("schema_valid")
		old_error = models.get("schema_error")
		try:
			models["schema_valid"] = False
			models["schema_error"] = "forced-test-mismatch"
			r = self.client.post("/api/score", json={"telemetry": {"speed": 10}})
			self.assertEqual(r.status_code, 503)
			data = r.get_json()
			self.assertEqual(data.get("error"), "schema_mismatch")
			self.assertIn("details", data)
		finally:
			models["schema_valid"] = old_valid
			models["schema_error"] = old_error

	def test_coach_contract(self):
		payload = {
			"session_id": "test-session",
			"score": 68,
			"features": {
				"proximity_score": 0.2,
				"braking_flag": 1,
				"lane_change_flag": 0,
			},
			"predicted_fuel_rate": 8.1,
			"history_summary": "mostly steady",
		}
		r = self.client.post("/api/coach", json=payload)
		self.assertEqual(r.status_code, 200)
		data = r.get_json()

		self.assertIn("message", data)
		self.assertIn("tips", data)
		self.assertIn("severity", data)
		self.assertIsInstance(data["tips"], list)
		self.assertGreaterEqual(len(data["tips"]), 2)
		self.assertLessEqual(len(data["tips"]), 3)
		self.assertIn(data["severity"], {"green", "yellow", "red"})

	def test_coach_generation_contract_bounds(self):
		payload = {
			"session_id": "coach-generation-bounds",
			"score": 58,
			"features": {
				"proximity_score": 0.18,
				"braking_flag": 1,
				"lane_change_flag": 1,
			},
			"predicted_fuel_rate": 8.6,
			"history_summary": "unstable in dense traffic",
		}
		r = self.client.post("/api/coach", json=payload)
		self.assertEqual(r.status_code, 200)
		data = r.get_json()

		self.assertIn("tips", data)
		self.assertIsInstance(data["tips"], list)
		self.assertGreaterEqual(len(data["tips"]), 2)
		self.assertLessEqual(len(data["tips"]), 3)
		for t in data["tips"]:
			self.assertIsInstance(t, str)
			self.assertTrue(t.strip())
			# Bounded length contract: concise coaching sentence.
			self.assertLessEqual(len(t), 220)
		self.assertIn(data.get("severity"), {"green", "yellow", "red"})

	def test_coach_generation_severity_alignment(self):
		# Green score should map to green severity.
		r_green = self.client.post(
			"/api/coach",
			json={"score": 82, "features": {}, "predicted_fuel_rate": 6.2},
		)
		self.assertEqual(r_green.status_code, 200)
		self.assertEqual(r_green.get_json().get("severity"), "green")

		# Yellow score should map to yellow severity.
		r_yellow = self.client.post(
			"/api/coach",
			json={"score": 62, "features": {}, "predicted_fuel_rate": 7.8},
		)
		self.assertEqual(r_yellow.status_code, 200)
		self.assertEqual(r_yellow.get_json().get("severity"), "yellow")

		# Red score should map to red severity.
		r_red = self.client.post(
			"/api/coach",
			json={"score": 35, "features": {}, "predicted_fuel_rate": 10.1},
		)
		self.assertEqual(r_red.status_code, 200)
		self.assertEqual(r_red.get_json().get("severity"), "red")

	def test_coach_rule_based_source_contract(self):
		"""Verify coaching always returns cv_rules source after Flan-T5 removal."""
		r = self.client.post(
			"/api/coach",
			json={
				"session_id": "coach-rules-branch",
				"score": 55,
				"features": {"braking_ratio": 1, "proximity_score": 0.2},
				"predicted_fuel_rate": 8.0,
			},
		)
		self.assertEqual(r.status_code, 200)
		data = r.get_json()
		self.assertEqual(data.get("source"), "cv_rules")
		self.assertIn("tips", data)
		self.assertGreaterEqual(len(data["tips"]), 2)
		self.assertLessEqual(len(data["tips"]), 3)
		self.assertTrue(data["tips"][0].strip())
		self.assertLessEqual(len(data["tips"][0]), 220)


if __name__ == "__main__":
	unittest.main(verbosity=2)
