#!/usr/bin/env python3
"""
Test script for Soil Cause Analyzer.

Tests:
1. basic analysis
2. analyze_prediction
3. determinism for same raw class
4. alias consistency for same canonical disease
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.soil_cause_analyzer import SoilCauseAnalyzer


def print_result(result: dict, disease_class: str) -> None:
    print(f"\n📊 Analysis for: {result['display_name']} ({disease_class})")
    print(f"   Healthy: {result['is_healthy']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Simulated: {result['simulated']}")
    print("   Current Values:")

    for param, value in result["current_values"].items():
        status = result["deviation_status"][param]
        healthy = result["healthy_ranges"][param]
        print(
            f"     {param}: {value} {healthy['unit']} "
            f"(status: {status}, healthy: {healthy['min']}-{healthy['max']})"
        )

    if result["contributing_factors"]:
        print("   Contributing Factors:")
        for factor in result["contributing_factors"]:
            print(f"     • {factor}")


def test_analyzer() -> None:
    analyzer = SoilCauseAnalyzer()

    test_classes = [
        "APPLE___blotch_apple",
        "APPLE___healthy_apple",
        "GUAVA___anthracnose_guava",
        "POMEGRANATE___alternaria_pomegranate",
    ]

    print("🧪 Testing Soil Cause Analyzer")
    print("=" * 50)

    for disease_class in test_classes:
        try:
            result = analyzer.analyze_disease_class(disease_class)
            if not result["success"]:
                print(f"❌ Analysis failed for {disease_class}: {result.get('error')}")
                continue
            print_result(result, disease_class)
        except Exception as e:
            print(f"❌ Error analyzing {disease_class}: {e}")

    print("\n" + "=" * 50)
    print("🧪 Testing analyze_prediction method")

    sample_prediction = {
        "disease_name": "APPLE___blotch_apple",
        "confidence": 0.95,
        "other_data": "ignored",
    }

    try:
        result = analyzer.analyze_prediction(sample_prediction)
        if result["success"]:
            print(f"✅ Prediction analysis successful for {result['display_name']}")
        else:
            print(f"❌ Prediction analysis failed: {result.get('error')}")
    except Exception as e:
        print(f"❌ Error in analyze_prediction: {e}")

    print("\n" + "=" * 50)
    print("🧪 Testing determinism (same raw class twice)")

    results1 = analyzer.analyze_disease_class("APPLE___blotch_apple")
    results2 = analyzer.analyze_disease_class("APPLE___blotch_apple")

    if results1["current_values"] == results2["current_values"]:
        print("✅ Deterministic results confirmed for same raw class")
    else:
        print("❌ Determinism failed for same raw class")
        print(f"   Run 1: {results1['current_values']}")
        print(f"   Run 2: {results2['current_values']}")

    print("\n" + "=" * 50)
    print("🧪 Testing alias consistency (same canonical disease)")

    alias1 = analyzer.analyze_disease_class("APPLE___scab_apple")
    alias2 = analyzer.analyze_disease_class("Apple___Apple_scab")

    print_result(alias1, "APPLE___scab_apple")
    print_result(alias2, "Apple___Apple_scab")

    if alias1["current_values"] == alias2["current_values"]:
        print("✅ Alias consistency confirmed for Apple Scab")
    else:
        print("❌ Alias consistency failed")
        print(f"   APPLE___scab_apple: {alias1['current_values']}")
        print(f"   Apple___Apple_scab: {alias2['current_values']}")

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    test_analyzer()
    