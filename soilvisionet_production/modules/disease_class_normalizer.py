"""
Disease Class Normalizer

This module provides utilities for normalizing disease class names from the
disease database into canonical forms for consistent use across the application.
"""

import json
import os
from typing import Dict, List, Any


def load_disease_database(path: str = "config/disease_database.json") -> Dict[str, Any]:
    """
    Load the disease database from JSON file.

    Args:
        path: Path to the disease database JSON file.

    Returns:
        Dictionary containing disease data keyed by raw class names.
    """
    with open(path, 'r') as f:
        return json.load(f)


def normalize_crop_name(crop: str) -> str:
    """
    Normalize crop name to canonical lowercase form.

    Args:
        crop: Raw crop name from database.

    Returns:
        Canonical crop name.
    """
    # Remove punctuation and lowercase
    normalized = crop.lower().replace('(', '').replace(')', '').replace(',', '').replace('_', ' ').strip()

    # Specific mappings for consistency
    mappings = {
        'cherry including sour': 'cherry',
        'corn maize': 'corn',
        'pepper bell': 'bell pepper',
    }

    return mappings.get(normalized, normalized)


def normalize_disease_name(condition: str, crop: str, is_disease: bool = True) -> str:
    """
    Normalize disease condition to canonical form.

    Args:
        condition: Raw condition string.
        crop: Canonical crop name for context.
        is_disease: Whether this is a disease (not healthy).

    Returns:
        Canonical disease name.
    """
    if not is_disease:
        return 'healthy'

    # Convert underscores to spaces
    normalized = condition.replace('_', ' ')

    # If condition ends with crop name, move crop to front
    crop_lower = crop.lower()
    if normalized.lower().endswith(' ' + crop_lower):
        disease_part = normalized[:-len(crop_lower) - 1].strip()
        normalized = f"{crop_lower} {disease_part}"

    # Collapse multiple spaces
    normalized = ' '.join(normalized.split())

    return normalized.lower()


def create_display_name(canonical_crop: str, canonical_disease: str, is_healthy: bool) -> str:
    """
    Create user-friendly display name.

    Args:
        canonical_crop: Canonical crop name.
        canonical_disease: Canonical disease name.
        is_healthy: Whether this is a healthy class.

    Returns:
        Display name for UI.
    """
    if is_healthy:
        return f"{canonical_crop.title()} Healthy"
    else:
        return canonical_disease.title()


def build_disease_class_mapping(disease_db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build the normalized mapping for all disease classes.

    Args:
        disease_db: Disease database dictionary.

    Returns:
        Mapping dictionary keyed by raw class names.
    """
    mapping = {}

    for raw_key, entry in disease_db.items():
        # Extract fields
        crop_raw = entry.get('crop', '')
        condition_raw = entry.get('condition', '')
        is_disease = entry.get('is_disease', True)

        # Normalize crop
        canonical_crop = normalize_crop_name(crop_raw)

        # Normalize disease
        canonical_disease = normalize_disease_name(condition_raw, canonical_crop, is_disease)

        # Determine if healthy
        is_healthy_class = not is_disease

        # Create display name
        display_name = create_display_name(canonical_crop, canonical_disease, is_healthy_class)

        # Create canonical key
        canonical_key = f"{canonical_crop}::{canonical_disease}"

        # Create aliases (deduplicated)
        aliases = list(set([
            raw_key,
            condition_raw,
            canonical_disease,
            display_name
        ]))

        # Build the mapping entry
        mapping[raw_key] = {
            "raw_key": raw_key,
            "canonical_crop": canonical_crop,
            "canonical_disease": canonical_disease,
            "display_name": display_name,
            "is_healthy_class": is_healthy_class,
            "raw_condition": condition_raw,
            "normalized_condition": canonical_disease,
            "canonical_key": canonical_key,
            "aliases": aliases
        }

    return mapping


def save_disease_class_mapping(mapping: Dict[str, Dict[str, Any]], path: str = "config/disease_class_mapping.json") -> None:
    """
    Save the disease class mapping to JSON file.

    Args:
        mapping: The mapping dictionary.
        path: Output file path.
    """
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)


def load_disease_class_mapping(path: str = "config/disease_class_mapping.json") -> Dict[str, Dict[str, Any]]:
    """
    Load the disease class mapping from JSON file.

    Args:
        path: Path to the mapping file.

    Returns:
        Mapping dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def normalize_disease_class(raw_key: str, mapping: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get normalized information for a disease class.

    Args:
        raw_key: Raw disease class key.
        mapping: Optional pre-loaded mapping. If None, loads from default path.

    Returns:
        Normalized disease class information.

    Raises:
        KeyError: If raw_key not found in mapping.
    """
    if mapping is None:
        mapping = load_disease_class_mapping()
    return mapping[raw_key]


def validate_mapping_coverage(disease_db: Dict[str, Any], mapping: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate that all disease classes in the database have mappings.

    Args:
        disease_db: Disease database.
        mapping: Disease class mapping.

    Returns:
        True if all classes are covered, False otherwise.
    """
    db_keys = set(disease_db.keys())
    mapping_keys = set(mapping.keys())
    return db_keys == mapping_keys


if __name__ == "__main__":
    # Build and save the mapping
    disease_db = load_disease_database()
    mapping = build_disease_class_mapping(disease_db)
    save_disease_class_mapping(mapping)

    # Validate
    if validate_mapping_coverage(disease_db, mapping):
        print("✓ All disease classes successfully mapped.")
    else:
        print("✗ Some disease classes are missing from mapping.")

    print(f"Generated mapping for {len(mapping)} disease classes.")