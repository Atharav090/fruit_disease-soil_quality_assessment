#!/usr/bin/env python3
"""
Build disease_soil_profiles.json from disease_class_mapping.json and crop_database.json.

Goals:
- Priority crops (apple, guava, pomegranate) are curated and reproducible
- No runtime dependence on risk_profile_override
- Healthy baseline entries for priority crops use real source metadata
- Non-priority crops fall back to generic demo-safe logic
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


SOIL_SCOPE = ["nitrogen", "phosphorus", "potassium", "ph"]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def expand_range(
    optimal: float,
    low_factor: float = 0.90,
    high_factor: float = 1.10,
    decimals: int = 2,
) -> tuple[float, float]:
    low = round(optimal * low_factor, decimals)
    high = round(optimal * high_factor, decimals)
    if low >= high:
        high = round(low + (0.1 if decimals > 0 else 1), decimals)
    return low, high


def default_optimal_soil() -> Dict[str, Dict[str, float]]:
    return {
        "nitrogen": {"min": 50, "max": 80, "optimal": 65},
        "phosphorus": {"min": 15, "max": 25, "optimal": 20},
        "potassium": {"min": 30, "max": 50, "optimal": 40},
        "ph": {"min": 6.0, "max": 7.0, "optimal": 6.5},
    }


def get_crop_optimal_soil(crop: str, crop_db: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    entry = crop_db.get(crop, {})
    optimal = entry.get("optimal_soil")
    if optimal:
        return optimal
    return default_optimal_soil()


def format_healthy_profile(
    optimal_soil: Dict[str, Dict[str, float]],
    crop: str,
    baseline_mode: str = "generic",
) -> Dict[str, Dict[str, Any]]:
    """
    Build healthy_profile as realistic ranges.

    baseline_mode:
    - generic: derive from optimal values
    - apple / guava / pomegranate: explicit tuned ranges
    """
    if baseline_mode == "apple":
        return {
            "nitrogen": {"min": 54.0, "max": 66.0, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "phosphorus": {"min": 18.0, "max": 22.0, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "potassium": {"min": 36.0, "max": 44.0, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "ph": {"min": 5.9, "max": 6.6, "unit": "pH", "value_origin": "source_backed"},
        }

    if baseline_mode == "guava":
        return {
            "nitrogen": {"min": 108.0, "max": 132.0, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "phosphorus": {"min": 40.5, "max": 49.5, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "potassium": {"min": 31.5, "max": 38.5, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "ph": {"min": 5.8, "max": 7.4, "unit": "pH", "value_origin": "source_backed"},
        }

    if baseline_mode == "pomegranate":
        return {
            "nitrogen": {"min": 67.5, "max": 82.5, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "phosphorus": {"min": 38.7, "max": 47.3, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "potassium": {"min": 23.4, "max": 28.6, "unit": "ppm", "value_origin": "derived_from_crop_baseline"},
            "ph": {"min": 7.4, "max": 8.0, "unit": "pH", "value_origin": "source_backed"},
        }

    healthy = {}
    for param, values in optimal_soil.items():
        unit = "pH" if param == "ph" else "ppm"
        if param == "ph":
            low, high = expand_range(values["optimal"], low_factor=0.95, high_factor=1.05, decimals=2)
        else:
            low, high = expand_range(values["optimal"], low_factor=0.90, high_factor=1.10, decimals=1)

        healthy[param] = {
            "min": low,
            "max": high,
            "unit": unit,
            "value_origin": "derived_from_crop_baseline",
        }
    return healthy


def derive_generic_risk_profile(optimal_soil: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    n_opt = optimal_soil["nitrogen"]["optimal"]
    p_opt = optimal_soil["phosphorus"]["optimal"]
    k_opt = optimal_soil["potassium"]["optimal"]
    ph_opt = optimal_soil["ph"]["optimal"]

    return {
        "nitrogen": {
            "direction": "high_or_imbalanced",
            "example_value": int(round(n_opt * 1.40)),
            "unit": "ppm",
            "value_origin": "derived_demo",
        },
        "phosphorus": {
            "direction": "low_or_imbalanced",
            "example_value": int(round(p_opt * 0.70)),
            "unit": "ppm",
            "value_origin": "derived_demo",
        },
        "potassium": {
            "direction": "low",
            "example_value": int(round(k_opt * 0.60)),
            "unit": "ppm",
            "value_origin": "derived_demo",
        },
        "ph": {
            "direction": "low_or_high",
            "example_value": round(ph_opt - 0.6, 1),
            "unit": "pH",
            "value_origin": "derived_demo",
        },
    }


def get_generic_causal_language(canonical_disease: str) -> str:
    disease = canonical_disease.lower()
    pathogen_indicators = ["virus", "fruitfly", "spider", "mite", "greening", "hlb"]
    if any(token in disease for token in pathogen_indicators):
        return "indirect_risk_associated"
    return "risk_associated"


def get_generic_contributing_parameters(is_healthy: bool) -> list[str]:
    if is_healthy:
        return []
    return [
        "Potassium imbalance",
        "Non-ideal soil pH",
        "Excess or imbalanced nitrogen",
    ]


PRIORITY_BASELINES: Dict[str, Dict[str, Any]] = {
    "apple": {
        "healthy_profile_mode": "apple",
        "source_notes": [
            "Apple crop baseline uses fruit-tree fertility guidance and orchard soil-pH guidance.",
            "Healthy orchard soil is modeled around moderately acidic to neutral conditions and balanced fertility."
        ],
        "sources": [
            {
                "title": "Penn State Extension - Soil Fertility for Fruit Trees",
                "domain": "extension.psu.edu",
                "evidence_use": "healthy_profile",
            },
            {
                "title": "University of Minnesota Extension - Growing Apples",
                "domain": "extension.umn.edu",
                "evidence_use": "crop_baseline",
            },
        ],
    },
    "guava": {
        "healthy_profile_mode": "guava",
        "source_notes": [
            "Guava crop baseline uses Indian university and horticulture guidance for well-drained soils.",
            "Healthy guava soil is modeled within a broad acceptable pH window and balanced fertility."
        ],
        "sources": [
            {
                "title": "TNAU - Guava Cultivation Practices",
                "domain": "tnau.ac.in",
                "evidence_use": "healthy_profile",
            },
            {
                "title": "National Horticulture Board - Guava",
                "domain": "nhb.gov.in",
                "evidence_use": "crop_baseline",
            },
        ],
    },
    "pomegranate": {
        "healthy_profile_mode": "pomegranate",
        "source_notes": [
            "Pomegranate crop baseline uses horticulture and IPM guidance for well-drained soils.",
            "Healthy pomegranate soil is modeled around slightly alkaline conditions where crop guidance supports it."
        ],
        "sources": [
            {
                "title": "National Horticulture Board - Pomegranate DPR",
                "domain": "nhb.gov.in",
                "evidence_use": "healthy_profile",
            },
            {
                "title": "UC IPM - Pomegranate",
                "domain": "ipm.ucanr.edu",
                "evidence_use": "crop_baseline",
            },
        ],
    },
}


PRIORITY_DISEASE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Apple
    "apple::apple blotch": {
        "contributing_parameters": [
            "Excess nitrogen can encourage dense foliage and prolong leaf wetness",
            "Soil pH outside the preferred orchard range can stress trees and weaken defenses",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Apple blotch is mainly associated with humid conditions and leaf wetness, not direct soil causation.",
            "The displayed soil readings are simulated risk-associated values derived from crop baseline and disease context.",
        ],
        "sources": [
            {
                "title": "Penn State Extension - Apple Disease Management",
                "domain": "extension.psu.edu",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "University of Minnesota Extension - Growing Apples",
                "domain": "extension.umn.edu",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 82, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 26, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 34, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 5.8, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "apple::apple rot": {
        "contributing_parameters": [
            "Poor drainage and soil moisture extremes increase rot risk",
            "Imbalanced nutrient levels, especially high nitrogen and low potassium, can reduce fruit firmness",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Apple fruit rot pressure is strongly linked with wet conditions and fruit stress.",
            "The displayed soil readings are simulated to represent stress-associated conditions, not direct causal thresholds.",
        ],
        "sources": [
            {
                "title": "Penn State Extension - Apple Nutrition",
                "domain": "extension.psu.edu",
                "evidence_use": "healthy_profile",
            },
            {
                "title": "University of Minnesota Extension - Black Rot of Apple",
                "domain": "extension.umn.edu",
                "evidence_use": "disease_relationship",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 86, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 27, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 30, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 6.0, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "apple::apple scab": {
        "contributing_parameters": [
            "High nitrogen promotes dense canopy and longer leaf wetness periods",
            "Low potassium may limit plant defense and recovery",
            "Non-ideal soil pH can contribute to orchard stress",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Apple scab severity is linked to foliage wetness and vigorous canopy growth rather than direct soil-causation.",
            "The displayed soil readings are simulated risk-associated values for demo use.",
        ],
        "sources": [
            {
                "title": "Penn State Extension - Apple Scab",
                "domain": "extension.psu.edu",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "University of Minnesota Extension - Growing Apples",
                "domain": "extension.umn.edu",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 84, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 28, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 32, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 6.1, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "apple::black rot": {
        "contributing_parameters": [
            "Wet conditions and poor air circulation encourage black rot spread",
            "Nutrient imbalance can reduce stress tolerance and fruit quality",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Black rot is linked to warm wet conditions and stressed trees.",
            "The displayed soil readings are simulated stress-associated readings derived from crop baseline.",
        ],
        "sources": [
            {
                "title": "University of Minnesota Extension - Black Rot of Apple",
                "domain": "extension.umn.edu",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "Penn State Extension - Soil Fertility for Fruit Trees",
                "domain": "extension.psu.edu",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 80, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 30, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 33, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 6.2, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "apple::cedar apple rust": {
        "contributing_parameters": [
            "Extended leaf wetness and humid conditions increase rust infection risk",
            "Soil fertility stress can reduce tree resistance, but soil is not a direct cause",
        ],
        "causal_language": "indirect_risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Cedar apple rust depends on alternate host presence and wet conditions.",
            "The displayed soil readings are indirect risk-style values intended for demo visualization.",
        ],
        "sources": [
            {
                "title": "University of Minnesota Extension - Cedar Apple Rust",
                "domain": "extension.umn.edu",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "University of Minnesota Extension - Growing Apples",
                "domain": "extension.umn.edu",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 78, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 25, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 33, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 6.2, "unit": "pH", "value_origin": "derived_demo"},
        },
    },

    # Guava
    "guava::guava anthracnose": {
        "contributing_parameters": [
            "High humidity and poor drainage favor anthracnose development",
            "Soil nutrient imbalance can stress trees and increase susceptibility",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Guava anthracnose is primarily a humidity-driven fungal disease.",
            "The displayed soil readings are simulated susceptibility-associated values derived from crop baseline.",
        ],
        "sources": [
            {
                "title": "TNAU - Guava Disease Management",
                "domain": "tnau.ac.in",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "TNAU - Guava Cultivation Practices",
                "domain": "tnau.ac.in",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 150, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 35, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 28, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 5.9, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "guava::guava fruitfly": {
        "contributing_parameters": [
            "Plant stress from nutrient imbalance can increase susceptibility to fruitfly damage",
            "Poor vigor from deficient potassium or phosphorus can reduce fruit resilience",
        ],
        "causal_language": "indirect_risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "low",
        "source_notes": [
            "Fruitfly damage is driven by insect pressure, not soil chemistry directly.",
            "The displayed soil readings are indirect risk-style values for demo use.",
        ],
        "sources": [
            {
                "title": "TNAU - Guava Fruit Fly Management",
                "domain": "tnau.ac.in",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "National Horticulture Board - Guava",
                "domain": "nhb.gov.in",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 140, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 40, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 25, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 5.8, "unit": "pH", "value_origin": "derived_demo"},
        },
    },

    # Pomegranate
    "pomegranate::pomegranate alternaria": {
        "contributing_parameters": [
            "Water stress and fruit cracking can raise alternaria fruit-rot risk",
            "Imbalanced nutrition can weaken plant resilience",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Alternaria fruit rot risk is associated with orchard stress and cracking conditions.",
            "The displayed soil readings are simulated risk-associated values derived from crop baseline.",
        ],
        "sources": [
            {
                "title": "UC IPM - Pomegranate Alternaria Fruit Rot",
                "domain": "ipm.ucanr.edu",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "National Horticulture Board - Pomegranate DPR",
                "domain": "nhb.gov.in",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 88, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 37, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 20, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 7.2, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "pomegranate::pomegranate anthracnose": {
        "contributing_parameters": [
            "Prolonged leaf wetness encourages anthracnose outbreaks",
            "Nutrient imbalance can increase susceptibility under humid conditions",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Pomegranate anthracnose management depends more on canopy aeration and moisture reduction than direct soil-causation.",
            "The displayed soil readings are simulated supportive-risk values.",
        ],
        "sources": [
            {
                "title": "TNAU - Major Fungicides and Crop Protection Notes",
                "domain": "tnau.ac.in",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "National Horticulture Board - Pomegranate DPR",
                "domain": "nhb.gov.in",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 87, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 39, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 24, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 7.1, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "pomegranate::pomegranate bacterial blight": {
        "contributing_parameters": [
            "Moisture-driven spread and plant stress favor bacterial blight pressure",
            "Balanced fertility supports resilience but is not a direct cause",
        ],
        "causal_language": "susceptibility_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "medium",
        "source_notes": [
            "Bacterial blight pressure is mainly weather and inoculum driven.",
            "The displayed soil readings are simulated susceptibility-associated values.",
        ],
        "sources": [
            {
                "title": "National Horticulture Board - Pomegranate Bacterial Blight Advisory",
                "domain": "nhb.gov.in",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "National Horticulture Board - Pomegranate DPR",
                "domain": "nhb.gov.in",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 83, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 36, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 23, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 7.0, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
    "pomegranate::pomegranate cercospora": {
        "contributing_parameters": [
            "Humid dense-canopy conditions favor cercospora leaf spot",
            "Balanced fertility supports foliage health and stress tolerance",
        ],
        "causal_language": "risk_associated",
        "evidence_type": "crop_baseline_plus_disease_group_override",
        "confidence": "low",
        "source_notes": [
            "Direct crop+disease-specific soil thresholds were not found for pomegranate cercospora.",
            "The displayed soil readings are low-confidence simulated values derived from crop baseline and disease family logic.",
        ],
        "sources": [
            {
                "title": "ICAR - Pomegranate Disease Notes",
                "domain": "icar.gov.in",
                "evidence_use": "disease_relationship",
            },
            {
                "title": "National Horticulture Board - Pomegranate DPR",
                "domain": "nhb.gov.in",
                "evidence_use": "healthy_profile",
            },
        ],
        "risk_profile": {
            "nitrogen": {"direction": "high_or_imbalanced", "example_value": 85, "unit": "ppm", "value_origin": "derived_demo"},
            "phosphorus": {"direction": "low_or_imbalanced", "example_value": 38, "unit": "ppm", "value_origin": "derived_demo"},
            "potassium": {"direction": "low", "example_value": 22, "unit": "ppm", "value_origin": "derived_demo"},
            "ph": {"direction": "low_or_high", "example_value": 7.1, "unit": "pH", "value_origin": "derived_demo"},
        },
    },
}


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def build_profile(
    raw_key: str,
    entry: Dict[str, Any],
    crop_db: Dict[str, Any],
) -> Dict[str, Any]:
    canonical_crop = entry["canonical_crop"]
    canonical_disease = entry["canonical_disease"]
    canonical_key = f"{canonical_crop}::{canonical_disease}"
    is_healthy = entry["is_healthy_class"]
    display_name = entry["display_name"]

    optimal_soil = get_crop_optimal_soil(canonical_crop, crop_db)
    baseline_meta = PRIORITY_BASELINES.get(canonical_crop)

    healthy_profile = format_healthy_profile(
        optimal_soil,
        canonical_crop,
        baseline_mode=baseline_meta["healthy_profile_mode"] if baseline_meta else "generic",
    )

    if baseline_meta:
        base_source_notes = deepcopy(baseline_meta["source_notes"])
        base_sources = deepcopy(baseline_meta["sources"])
    else:
        base_source_notes = [
            "Crop-level optimal soil values used for baseline.",
            "Values derived from crop database optimal_soil.",
        ]
        base_sources = [
            {
                "title": "Crop Database Baseline",
                "domain": "internal_crop_database",
                "evidence_use": "healthy_profile",
            }
        ]

    profile: Dict[str, Any] = {
        "raw_key": raw_key,
        "canonical_crop": canonical_crop,
        "canonical_disease": canonical_disease,
        "display_name": display_name,
        "is_healthy_class": is_healthy,
        "soil_parameter_scope": SOIL_SCOPE,
        "healthy_profile": healthy_profile,
        "contributing_parameters": [],
        "causal_language": "",
        "reading_mode": "baseline" if is_healthy else "simulated",
        "evidence_type": "crop_baseline",
        "confidence": "high" if is_healthy else "low",
        "source_notes": base_source_notes,
        "sources": base_sources,
        "risk_profile": {},
    }

    if is_healthy:
        return profile

    profile["risk_profile"] = derive_generic_risk_profile(optimal_soil)
    profile["contributing_parameters"] = get_generic_contributing_parameters(False)
    profile["causal_language"] = get_generic_causal_language(canonical_disease)
    profile["evidence_type"] = "crop_baseline_plus_disease_risk"
    profile["confidence"] = "low"
    profile["source_notes"] = [
        "Exact disease-specific NPK values were not found in authoritative sources for this class.",
        "Crop-level soil guidance is used as baseline.",
        "Displayed disease-side soil readings are conservative simulated demo values.",
        "Soil parameters are risk-associated, not direct causes.",
    ]
    profile["sources"] = base_sources

    disease_override = PRIORITY_DISEASE_OVERRIDES.get(canonical_key)
    if disease_override:
        profile = merge_dict(profile, disease_override)

    return profile


def build_soil_profiles() -> None:
    base_path = Path(__file__).resolve().parent.parent
    mapping_path = base_path / "config" / "disease_class_mapping.json"
    crop_db_path = base_path / "config" / "crop_database.json"
    output_path = base_path / "config" / "disease_soil_profiles.json"

    mapping = load_json(mapping_path)
    crop_db = load_json(crop_db_path)

    profiles: Dict[str, Any] = {}
    for raw_key, entry in mapping.items():
        profiles[raw_key] = build_profile(raw_key, entry, crop_db)

    save_json(output_path, profiles)
    print(f"Generated {len(profiles)} soil profiles at {output_path}")


if __name__ == "__main__":
    build_soil_profiles()