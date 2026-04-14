#!/usr/bin/env python3
"""
Test script for UI integration of Soil Cause Analyzer.

This verifies the same safe integration pattern used in ui/app.py:
- analyzer is called after image prediction
- result is enriched with:
    - soil_factor_analysis
    - simulated_sensor_readings
- failures do not crash the flow
"""

import sys
from pathlib import Path

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.soil_cause_analyzer import SoilCauseAnalyzer


def print_soil_analysis(analysis: dict) -> None:
    """Pretty-print enriched soil analysis."""
    print(f"\n📈 Soil Analysis for: {analysis['display_name']}")
    print(f"   Healthy: {analysis['is_healthy']}")
    print(f"   Simulated: {analysis['simulated']}")
    print("   Current Values:")

    for param, reading in analysis["current_readings"].items():
        value = reading["value"]
        unit = reading["unit"]
        status = reading["status"]
        hr = reading["healthy_range"]
        print(
            f"     {param}: {value} {unit} "
            f"(status: {status}, healthy: {hr['min']}-{hr['max']})"
        )

    factors = analysis.get("contributing_factors", [])
    if factors:
        print("   Contributing Factors:")
        for factor in factors:
            print(f"     • {factor}")


def integrate_soil_analysis(result: dict, analyzer: SoilCauseAnalyzer) -> dict:
    """
    Mimic the exact safe integration logic used in ui/app.py.
    """
    soil_analysis = None
    simulated_sensor_readings = {}

    if result.get("success", False) and "primary_prediction" in result:
        try:
            analysis = analyzer.analyze_prediction(result["primary_prediction"])

            if analysis.get("success"):
                soil_analysis = analysis
                simulated_sensor_readings = analysis.get("current_readings", {})
            else:
                soil_analysis = None
                simulated_sensor_readings = {}

        except Exception as soil_err:
            print(f"⚠️ Soil analysis failed safely: {soil_err}")
            soil_analysis = None
            simulated_sensor_readings = {}

    result["soil_factor_analysis"] = soil_analysis
    result["simulated_sensor_readings"] = simulated_sensor_readings
    return result


def test_basic_integration(analyzer: SoilCauseAnalyzer) -> bool:
    """Test enrichment on a mock disease detection result."""
    mock_result = {
        "success": True,
        "model_used": "demo",
        "primary_prediction": {
            "disease_name": "APPLE___blotch_apple",
            "confidence_percent": 85.5,
            "crop": "apple",
            "condition": "blotch",
            "is_disease": True,
            "severity": "moderate",
        },
        "top_predictions": [],
        "image_shape": (224, 224, 3),
        "all_probabilities": [],
        "recommendations": {},
    }

    print("📊 Mock detection result created")

    enriched = integrate_soil_analysis(mock_result, analyzer)

    assert "soil_factor_analysis" in enriched
    assert "simulated_sensor_readings" in enriched

    analysis = enriched["soil_factor_analysis"]
    readings = enriched["simulated_sensor_readings"]

    assert analysis is not None, "soil_factor_analysis should not be None"
    assert analysis.get("success") is True, "soil_factor_analysis should be successful"
    assert readings == analysis.get(
        "current_readings", {}
    ), "simulated_sensor_readings must match current_readings"

    print("✅ Soil analysis integrated into result")
    print("✅ Result enrichment verified")
    print_soil_analysis(analysis)
    return True


def test_failure_safety(analyzer: SoilCauseAnalyzer) -> bool:
    """Test that missing/unknown class does not crash integration."""
    mock_result = {
        "success": True,
        "primary_prediction": {
            "disease_name": "UNKNOWN___fake_class"
        },
    }

    enriched = integrate_soil_analysis(mock_result, analyzer)

    assert enriched["soil_factor_analysis"] is None
    assert enriched["simulated_sensor_readings"] == {}

    print("✅ Failure safety verified for unknown class")
    return True


def test_no_primary_prediction(analyzer: SoilCauseAnalyzer) -> bool:
    """Test that missing primary_prediction does not crash integration."""
    mock_result = {
        "success": True
    }

    enriched = integrate_soil_analysis(mock_result, analyzer)

    assert enriched["soil_factor_analysis"] is None
    assert enriched["simulated_sensor_readings"] == {}

    print("✅ Safe handling verified when primary_prediction is missing")
    return True


def test_unsuccessful_detection(analyzer: SoilCauseAnalyzer) -> bool:
    """Test that unsuccessful detection result is left safely un-enriched."""
    mock_result = {
        "success": False,
        "error": "Mock detection failure"
    }

    enriched = integrate_soil_analysis(mock_result, analyzer)

    assert enriched["soil_factor_analysis"] is None
    assert enriched["simulated_sensor_readings"] == {}

    print("✅ Safe handling verified for unsuccessful detection result")
    return True


def test_consistency(analyzer: SoilCauseAnalyzer) -> bool:
    """Test consistent enrichment for same disease class."""
    mock_result_1 = {
        "success": True,
        "primary_prediction": {"disease_name": "APPLE___blotch_apple"},
    }
    mock_result_2 = {
        "success": True,
        "primary_prediction": {"disease_name": "APPLE___blotch_apple"},
    }

    enriched_1 = integrate_soil_analysis(mock_result_1, analyzer)
    enriched_2 = integrate_soil_analysis(mock_result_2, analyzer)

    assert enriched_1["simulated_sensor_readings"] == enriched_2["simulated_sensor_readings"]
    print("✅ Deterministic UI enrichment verified")
    return True


def main() -> int:
    print("🧪 Testing UI Integration: Disease Detection + Soil Analysis")
    print("=" * 60)

    try:
        analyzer = SoilCauseAnalyzer()
        print("✅ SoilCauseAnalyzer initialized")

        test_basic_integration(analyzer)
        print()

        test_failure_safety(analyzer)
        test_no_primary_prediction(analyzer)
        test_unsuccessful_detection(analyzer)
        test_consistency(analyzer)

        print("\n🎉 Integration test passed!")
        return 0

    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())