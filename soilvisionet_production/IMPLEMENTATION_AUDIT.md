# Step 1 Audit - Image Prediction Flow

## Current Flow
ui/app.py
-> DiseaseDetector.detect_from_array(...)
-> DiseaseDetector._run_detection(...)
-> InferenceEngine.predict_vit(...)
-> DiseaseDetector builds primary_prediction/top_predictions
-> ui/app.py receives result
-> ExplanationGenerator.explain_detection(result)

## Best Injection Point
modules/disease_detector.py in _run_detection(), after:
    primary = top_predictions[0]

and before:
    return { ... }

## Reason
This point has the final disease prediction and metadata, so soil-parameter enrichment can be attached once and reused by UI, reports, and explanation modules.

## No changes to model inference
Do not modify InferenceEngine.predict_vit() for soil enrichment.
