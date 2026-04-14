#!/usr/bin/env python3
"""
Validation script for disease class mapping.

This script validates that the disease class mapping covers all classes
in the disease database and checks for any issues.
"""

import json
import sys
from pathlib import Path

def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def validate_mapping():
    """Validate the disease class mapping."""
    base_path = Path(__file__).parent.parent

    # Load files
    disease_db_path = base_path / "config" / "disease_database.json"
    mapping_path = base_path / "config" / "disease_class_mapping.json"

    if not disease_db_path.exists():
        print(f"✗ Disease database not found: {disease_db_path}")
        return False

    if not mapping_path.exists():
        print(f"✗ Disease class mapping not found: {mapping_path}")
        return False

    disease_db = load_json(disease_db_path)
    mapping = load_json(mapping_path)

    # Check coverage
    db_keys = set(disease_db.keys())
    mapping_keys = set(mapping.keys())

    if db_keys != mapping_keys:
        missing = db_keys - mapping_keys
        extra = mapping_keys - db_keys
        if missing:
            print(f"✗ Missing mappings for: {sorted(missing)}")
        if extra:
            print(f"✗ Extra mappings for: {sorted(extra)}")
        return False

    print(f"✓ All {len(db_keys)} disease classes are mapped.")

    # Check for duplicates in canonical keys
    canonical_keys = {}
    duplicates = []
    for raw_key, entry in mapping.items():
        ck = entry['canonical_key']
        if ck in canonical_keys:
            duplicates.append((ck, raw_key, canonical_keys[ck]))
        else:
            canonical_keys[ck] = raw_key

    if duplicates:
        print("⚠ Duplicate canonical keys found:")
        for ck, key1, key2 in duplicates:
            print(f"  {ck}: {key1}, {key2}")
    else:
        print("✓ No duplicate canonical keys.")

    # Summary
    healthy_count = sum(1 for entry in mapping.values() if entry['is_healthy_class'])
    disease_count = len(mapping) - healthy_count
    print(f"✓ Mapped {disease_count} disease classes and {healthy_count} healthy classes.")

    return True

if __name__ == "__main__":
    success = validate_mapping()
    sys.exit(0 if success else 1)