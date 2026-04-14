# Disease Soil Research Notes

## Overview
This document summarizes the research and derivation process for `config/disease_soil_profiles.json`.

## Methodology
- **Source Priority**: Internal crop database baselines used as foundation
- **Evidence Quality**: All entries marked as "derived_demo" or "derived_from_crop_baseline"
- **Confidence Levels**: All disease entries set to "low" confidence due to lack of authoritative disease-specific soil research
- **Causal Language**: "risk_associated" for fungal/bacterial diseases, "indirect_risk_associated" for viruses/pests

## Classes with Direct Evidence
None - No authoritative sources found linking specific soil NPK/pH values to disease causation.

## Classes Using Crop Baseline Fallback
All 39 diseased classes use crop-level optimal soil values as healthy baseline.

## Classes with Low Confidence (Need Manual Review)
All disease entries are low confidence and use derived risk profiles:
- Risk profiles created by adjusting optimal values (e.g., high N for fungal diseases, low K for stress susceptibility)
- Values are demonstrative only and not based on peer-reviewed evidence

## Pathogen-Specific Handling
- **Viruses** (e.g., Tomato Yellow Leaf Curl Virus): indirect_risk_associated
- **Pests** (e.g., fruitfly, spider mites): indirect_risk_associated
- **Fungal/Bacterial** (e.g., scab, rot): risk_associated

## Data Sources
- Primary: `config/crop_database.json` for crop baselines
- Secondary: Conservative derivation for risk profiles
- No external research sources accessed due to implementation constraints

## Recommendations for Future Enhancement
1. Consult agricultural extension services (e.g., University of Minnesota, Cornell, TNAU)
2. Review peer-reviewed papers on soil-disease relationships
3. Focus on well-studied diseases like apple scab, tomato blight
4. Consider soil microbiology factors beyond NPK/pH

## Coverage Summary
- Total classes: 55 (39 diseased + 16 healthy)
- All classes covered with complete soil profiles
- All entries include required metadata fields
- Validation passed for all profiles