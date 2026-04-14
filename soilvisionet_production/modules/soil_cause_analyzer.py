#!/usr/bin/env python3
"""
Soil Cause Analyzer

Generates deterministic simulated soil readings from disease soil profiles.
The same canonical disease always produces the same readings, even if there
are multiple raw dataset class aliases for it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class SoilCauseAnalyzer:
    """Analyze disease classes and generate deterministic simulated soil readings."""

    def __init__(self, soil_profiles_path: Optional[str] = None):
        if soil_profiles_path is None:
            soil_profiles_path = (
                Path(__file__).resolve().parent.parent
                / "config"
                / "disease_soil_profiles.json"
            )
        else:
            soil_profiles_path = Path(soil_profiles_path)

        self.soil_profiles_path = Path(soil_profiles_path)
        self.soil_profiles: Dict[str, Any] = self._load_profiles()

    def _load_profiles(self) -> Dict[str, Any]:
        if not self.soil_profiles_path.exists():
            raise FileNotFoundError(
                f"Soil profiles file not found: {self.soil_profiles_path}"
            )

        with open(self.soil_profiles_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _stable_unit_float(self, key: str) -> float:
        """
        Return a deterministic float in [0, 1) for a given key.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16**12)

    def _canonical_seed_key(self, profile: Dict[str, Any]) -> str:
        """
        Use canonical disease identity, not raw dataset key,
        so aliases like APPLE___scab_apple and Apple___Apple_scab
        produce the same readings.
        """
        crop = str(profile.get("canonical_crop", "")).strip().lower()
        disease = str(profile.get("canonical_disease", "")).strip().lower()
        return f"{crop}::{disease}"

    def _get_status(self, value: float, min_value: float, max_value: float) -> str:
        if value < min_value:
            return "low"
        if value > max_value:
            return "high"
        return "optimal"

    def _clamp_value(self, param: str, value: float) -> float:
        if param == "ph":
            return max(3.5, min(9.5, value))
        return max(0.0, value)

    def _round_value(self, param: str, value: float) -> float:
        if param == "ph":
            return round(value, 2)
        return round(value, 2)

    def _simulate_healthy_value(
        self,
        seed_key: str,
        param: str,
        min_v: float,
        max_v: float,
    ) -> float:
        """
        Generate a deterministic healthy reading inside the healthy range.
        """
        span = max_v - min_v
        midpoint = (min_v + max_v) / 2.0
        u = self._stable_unit_float(f"{seed_key}:{param}:healthy")

        # Keep healthy values comfortably inside the range.
        jitter = (u - 0.5) * span * 0.35
        value = midpoint + jitter

        # Safety clamp to stay in range.
        margin = span * 0.05
        lower_bound = min_v + margin
        upper_bound = max_v - margin

        if lower_bound <= upper_bound:
            value = max(lower_bound, min(upper_bound, value))
        else:
            value = max(min_v, min(max_v, value))

        value = self._clamp_value(param, value)
        return self._round_value(param, value)

    def _simulate_disease_value(
        self,
        seed_key: str,
        param: str,
        min_v: float,
        max_v: float,
        risk_info: Dict[str, Any],
    ) -> float:
        """
        Generate a deterministic disease-associated reading using the risk profile
        as the anchor, with small stable variation.
        """
        span = max_v - min_v
        example = float(risk_info.get("example_value", (min_v + max_v) / 2.0))
        direction = str(risk_info.get("direction", "")).strip().lower()

        u = self._stable_unit_float(f"{seed_key}:{param}:disease")

        # Small deterministic jitter around the example value
        jitter_scale = 0.04 if param == "ph" else 0.05
        value = example * (1 + ((u - 0.5) * 2 * jitter_scale))

        # Enforce abnormal side if needed
        out_margin = max(span * 0.08, 0.1)

        if direction == "low":
            value = min(value, min_v - out_margin)
        elif direction == "high":
            value = max(value, max_v + out_margin)
        elif direction == "high_or_imbalanced":
            value = max(value, max_v + out_margin)
        elif direction == "low_or_imbalanced":
            if example < min_v:
                value = min(value, min_v - out_margin)
            elif example > max_v:
                value = max(value, max_v + out_margin)
            else:
                value = min(value, min_v - out_margin)
        elif direction == "low_or_high":
            if example < min_v:
                value = min(value, min_v - out_margin)
            elif example > max_v:
                value = max(value, max_v + out_margin)
            else:
                if u < 0.5:
                    value = min(value, min_v - out_margin)
                else:
                    value = max(value, max_v + out_margin)

        value = self._clamp_value(param, value)
        return self._round_value(param, value)

    def _build_reading(
        self,
        seed_key: str,
        param: str,
        healthy_info: Dict[str, Any],
        risk_info: Optional[Dict[str, Any]],
        is_healthy: bool,
    ) -> Dict[str, Any]:
        min_v = float(healthy_info["min"])
        max_v = float(healthy_info["max"])
        unit = healthy_info.get("unit", "ppm")

        if is_healthy or not risk_info:
            value = self._simulate_healthy_value(seed_key, param, min_v, max_v)
            direction_hint = "optimal"
        else:
            value = self._simulate_disease_value(seed_key, param, min_v, max_v, risk_info)
            direction_hint = risk_info.get("direction", "")

        status = self._get_status(value, min_v, max_v)

        return {
            "value": value,
            "unit": unit,
            "healthy_range": {
                "min": min_v,
                "max": max_v,
                "unit": unit,
            },
            "status": status,
            "direction_hint": direction_hint,
        }

    def analyze_disease_class(self, disease_class: str) -> Dict[str, Any]:
        profile = self.soil_profiles.get(disease_class)

        if not profile:
            return {
                "success": False,
                "disease_class": disease_class,
                "error": f"No soil profile found for disease class: {disease_class}",
                "simulated": False,
            }

        is_healthy = bool(profile.get("is_healthy_class", False))
        healthy_profile = profile.get("healthy_profile", {})
        risk_profile = profile.get("risk_profile", {})
        seed_key = self._canonical_seed_key(profile)

        current_values: Dict[str, float] = {}
        healthy_ranges: Dict[str, Dict[str, Any]] = {}
        deviation_status: Dict[str, str] = {}
        current_readings: Dict[str, Dict[str, Any]] = {}

        for param, healthy_info in healthy_profile.items():
            risk_info = None if is_healthy else risk_profile.get(param)
            reading = self._build_reading(
                seed_key=seed_key,
                param=param,
                healthy_info=healthy_info,
                risk_info=risk_info,
                is_healthy=is_healthy,
            )

            current_readings[param] = reading
            current_values[param] = reading["value"]
            healthy_ranges[param] = reading["healthy_range"]
            deviation_status[param] = reading["status"]

        return {
            "success": True,
            "disease_class": disease_class,
            "canonical_crop": profile.get("canonical_crop"),
            "canonical_disease": profile.get("canonical_disease"),
            "display_name": profile.get("display_name"),
            "is_healthy": is_healthy,
            "simulated": True,
            "reading_mode": profile.get("reading_mode", "simulated"),
            "confidence": profile.get("confidence"),
            "evidence_type": profile.get("evidence_type"),
            "current_values": current_values,
            "healthy_ranges": healthy_ranges,
            "deviation_status": deviation_status,
            "current_readings": current_readings,
            "contributing_factors": profile.get("contributing_parameters", []),
            "source_notes": profile.get("source_notes", []),
            "disclaimer": (
                "These are simulated field-style sensor readings generated for demo "
                "purposes from the disease soil profile."
            ),
            "profile_used": {
                "is_healthy_class": is_healthy
            },
        }

    def analyze_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        disease_class = prediction.get("disease_name")
        if not disease_class:
            return {
                "success": False,
                "error": "Prediction dictionary does not contain 'disease_name'.",
                "simulated": False,
            }

        return self.analyze_disease_class(disease_class)