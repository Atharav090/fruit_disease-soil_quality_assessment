#!/usr/bin/env python3
"""
Validate disease_soil_profiles.json.

Focus:
- full coverage
- schema consistency
- priority crop research-quality checks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


PRIORITY_RAW_KEYS = {
    # Apple
    "APPLE___blotch_apple",
    "APPLE___healthy_apple",
    "APPLE___rot_apple",
    "APPLE___scab_apple",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    # Guava
    "GUAVA___anthracnose_guava",
    "GUAVA___fruitfly_guava",
    "GUAVA___healthy_guava",
    # Pomegranate
    "POMEGRANATE___alternaria_pomegranate",
    "POMEGRANATE___anthracnose_pomegranate",
    "POMEGRANATE___bacterial_blight_pomegranate",
    "POMEGRANATE___cercospora_pomegranate",
    "POMEGRANATE___healthy_pomegranate",
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_profile(raw_key: str, profile: Dict[str, Any], issues: list[str]) -> None:
    required_fields = [
        "raw_key",
        "canonical_crop",
        "canonical_disease",
        "display_name",
        "is_healthy_class",
        "soil_parameter_scope",
        "healthy_profile",
        "risk_profile",
        "contributing_parameters",
        "reading_mode",
        "evidence_type",
        "confidence",
        "source_notes",
        "sources",
    ]
    valid_confidences = {"high", "medium", "low"}

    missing_fields = [f for f in required_fields if f not in profile]
    if missing_fields:
        issues.append(f"{raw_key}: missing fields {missing_fields}")

    if profile.get("confidence") not in valid_confidences:
        issues.append(f"{raw_key}: invalid confidence '{profile.get('confidence')}'")

    is_healthy = profile.get("is_healthy_class", False)
    scope = profile.get("soil_parameter_scope", [])
    healthy = profile.get("healthy_profile", {})
    risk = profile.get("risk_profile", {})

    for param in scope:
        if param not in healthy:
            issues.append(f"{raw_key}: missing {param} in healthy_profile")
        if not is_healthy and param not in risk:
            issues.append(f"{raw_key}: missing {param} in risk_profile")

    if is_healthy:
        if profile.get("contributing_parameters"):
            issues.append(f"{raw_key}: healthy class should have empty contributing_parameters")
        if profile.get("causal_language"):
            issues.append(f"{raw_key}: healthy class should have empty causal_language")
        if risk:
            issues.append(f"{raw_key}: healthy class should have empty risk_profile")
    else:
        if not profile.get("contributing_parameters"):
            issues.append(f"{raw_key}: diseased class should have contributing_parameters")
        if not profile.get("causal_language"):
            issues.append(f"{raw_key}: diseased class should have causal_language")

    # healthy range sanity
    for param, values in healthy.items():
        if "min" in values and "max" in values:
            if values["min"] >= values["max"]:
                issues.append(f"{raw_key}: healthy_profile[{param}] min must be < max")

    # sources sanity
    sources = profile.get("sources", [])
    if not sources:
        issues.append(f"{raw_key}: missing sources")

    for i, src in enumerate(sources):
        if not src.get("title"):
            issues.append(f"{raw_key}: source[{i}] missing title")
        if not src.get("domain"):
            issues.append(f"{raw_key}: source[{i}] missing domain")
        if not src.get("evidence_use"):
            issues.append(f"{raw_key}: source[{i}] missing evidence_use")

    if not profile.get("source_notes"):
        issues.append(f"{raw_key}: missing source_notes")

    if "risk_profile_override" in profile:
        issues.append(f"{raw_key}: risk_profile_override must not exist in final runtime JSON")


def validate_priority_rules(profiles: Dict[str, Any], issues: list[str]) -> None:
    missing_priority = PRIORITY_RAW_KEYS - set(profiles.keys())
    if missing_priority:
        issues.append(f"Missing priority profiles: {sorted(missing_priority)}")

    # Healthy priority entries must not use placeholder-only sources
    for raw_key in PRIORITY_RAW_KEYS:
        profile = profiles.get(raw_key)
        if not profile:
            continue

        sources = profile.get("sources", [])
        domains = {src.get("domain", "") for src in sources}
        is_healthy = profile.get("is_healthy_class", False)

        if is_healthy and "internal_crop_database" in domains:
            issues.append(f"{raw_key}: priority healthy profile must not use internal_crop_database")

        if not is_healthy and all(d == "internal_crop_database" for d in domains if d):
            issues.append(f"{raw_key}: priority diseased profile must include real source domains")

    # Specific causal language rules
    fruitfly = profiles.get("GUAVA___fruitfly_guava", {})
    if fruitfly.get("causal_language") != "indirect_risk_associated":
        issues.append("GUAVA___fruitfly_guava: causal_language must be indirect_risk_associated")

    cedar = profiles.get("Apple___Cedar_apple_rust", {})
    if cedar.get("causal_language") != "indirect_risk_associated":
        issues.append("Apple___Cedar_apple_rust: causal_language must be indirect_risk_associated")

    # Duplicate scab variants may match each other, but not all apple diseases
    apple_compare_keys = [
        "APPLE___blotch_apple",
        "APPLE___rot_apple",
        "APPLE___scab_apple",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
    ]
    apple_risks = []
    for key in apple_compare_keys:
        if key in profiles:
            apple_risks.append(json.dumps(profiles[key]["risk_profile"], sort_keys=True))
    if len(set(apple_risks)) < 4:
        issues.append("Priority apple diseased profiles are still too repetitive")

    # Guava anthracnose and fruitfly must differ
    if (
        "GUAVA___anthracnose_guava" in profiles
        and "GUAVA___fruitfly_guava" in profiles
        and json.dumps(profiles["GUAVA___anthracnose_guava"]["risk_profile"], sort_keys=True)
        == json.dumps(profiles["GUAVA___fruitfly_guava"]["risk_profile"], sort_keys=True)
    ):
        issues.append("Guava anthracnose and guava fruitfly must not share identical risk_profile")

    # Pomegranate diseases should not all be clones
    pom_keys = [
        "POMEGRANATE___alternaria_pomegranate",
        "POMEGRANATE___anthracnose_pomegranate",
        "POMEGRANATE___bacterial_blight_pomegranate",
        "POMEGRANATE___cercospora_pomegranate",
    ]
    pom_risks = []
    for key in pom_keys:
        if key in profiles:
            pom_risks.append(json.dumps(profiles[key]["risk_profile"], sort_keys=True))
    if len(set(pom_risks)) < 3:
        issues.append("Priority pomegranate diseased profiles are still too repetitive")


def main() -> int:
    base_path = Path(__file__).resolve().parent.parent
    mapping_path = base_path / "config" / "disease_class_mapping.json"
    profiles_path = base_path / "config" / "disease_soil_profiles.json"

    if not mapping_path.exists():
        print(f"✗ Missing mapping file: {mapping_path}")
        return 1
    if not profiles_path.exists():
        print(f"✗ Missing profiles file: {profiles_path}")
        return 1

    mapping = load_json(mapping_path)
    profiles = load_json(profiles_path)

    issues: list[str] = []

    mapping_keys = set(mapping.keys())
    profile_keys = set(profiles.keys())
    if mapping_keys != profile_keys:
        missing = sorted(mapping_keys - profile_keys)
        extra = sorted(profile_keys - mapping_keys)
        if missing:
            issues.append(f"Missing profile keys: {missing}")
        if extra:
            issues.append(f"Extra profile keys: {extra}")

    for raw_key, profile in profiles.items():
        validate_profile(raw_key, profile, issues)

    validate_priority_rules(profiles, issues)

    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(f"✓ Validated {len(profiles)} soil profiles successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())