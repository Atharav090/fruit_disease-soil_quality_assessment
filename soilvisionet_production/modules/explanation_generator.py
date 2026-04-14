"""
Explanation Generator
Generates human-readable explanations and recommendations for predictions and assessments
"""

import json
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generate natural language explanations for predictions"""
    
    # Disease severity templates
    SEVERITY_EXPLANATIONS = {
        'mild': {
            'description': 'Minor disease symptoms that may not significantly impact yield',
            'urgency': 'Low priority, but monitor closely',
            'management': 'Maintain good cultural practices and monitor for progression'
        },
        'moderate': {
            'description': 'Disease symptoms present with potential to reduce yield',
            'urgency': 'Medium priority, intervention recommended',
            'management': 'Implement field sanitation and consider treatment options'
        },
        'severe': {
            'description': 'Significant disease symptoms with high yield loss potential',
            'urgency': 'High priority, immediate action required',
            'management': 'Apply treatment immediately and increase monitoring'
        }
    }
    
    @staticmethod
    def explain_detection(detection_result: Dict) -> Dict:
        """
        Generate explanation for disease detection result
        
        Args:
            detection_result: Output from DiseaseDetector
        
        Returns:
            Explanation dictionary with user-friendly text
        """
        if not detection_result.get('success', False):
            return {
                'status': 'Error',
                'message': detection_result.get('error', 'Detection failed'),
                'details': None
            }
        
        prediction = detection_result.get('primary_prediction', {})
        
        disease_name = prediction.get('disease_name', 'Unknown')
        crop = prediction.get('crop', 'Unknown crop')
        condition = prediction.get('condition', 'Unknown condition')
        confidence = prediction.get('confidence_percent', 0)
        is_disease = prediction.get('is_disease', False)
        severity = prediction.get('severity', 'moderate')
        
        # Main explanation
        if is_disease:
            main_text = f"""
DETECTION RESULT: Disease Identified
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Disease: {disease_name.replace('_', ' ').title()}
Crop Affected: {crop.replace('_', ' ').title()}
Condition: {condition.replace('_', ' ').title()}
Confidence: {confidence:.1f}%
Severity Level: {severity.upper()}

WHAT THIS MEANS:
Your crop has been identified as affected by {disease_name.replace('_', ' ').lower()}.
The model is {confidence:.1f}% confident in this identification.

{ExplanationGenerator.SEVERITY_EXPLANATIONS.get(severity, {}).get('description', 'Monitor the condition.')}
"""
        else:
            main_text = f"""
DETECTION RESULT: Plant Health Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: {condition.upper()}
Crop: {crop.replace('_', ' ').title()}
Confidence: {confidence:.1f}%

INTERPRETATION:
Your {crop.replace('_', ' ')} plant appears to be healthy based on the analyzed image.
No significant disease symptoms were detected.
Continue with regular monitoring and management practices.
"""
        
        # Symptoms explanation
        symptoms = prediction.get('symptoms', {})
        symptoms_text = ""
        if symptoms and is_disease:
            symptoms_text = f"""
SYMPTOMS TO LOOK FOR:
━━━━━━━━━━━━━━━━━━━

Visual Signs: {symptoms.get('visual', 'N/A')}
Tissue Damage: {symptoms.get('tissue_damage', 'N/A')}

Prevention: {symptoms.get('prevention', 'N/A')}
"""
        
        # Treatment explanation
        treatment = prediction.get('treatment', {})
        treatment_text = ""
        if treatment and is_disease:
            treatment_text = f"""
MANAGEMENT RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━

Immediate Actions:
  {treatment.get('immediate', 'Monitor the condition')}

Chemical Options:
  {treatment.get('chemical', 'Consult local extension officer for approved treatments')}

Ongoing Management:
  {treatment.get('management', 'Monitor and adjust as needed')}
"""
        
        # Urgency
        urgency = ExplanationGenerator.SEVERITY_EXPLANATIONS.get(severity, {}).get('urgency', 'Monitor')
        urgency_text = f"\n⚠️  URGENCY LEVEL: {urgency}\n"
        
        full_explanation = main_text + symptoms_text + treatment_text + urgency_text
        
        return {
            'status': 'Success',
            'main_explanation': main_text,
            'symptoms_explanation': symptoms_text,
            'treatment_explanation': treatment_text,
            'urgency': urgency,
            'full_explanation': full_explanation,
            'confidence_analysis': ExplanationGenerator._explain_confidence(
                confidence,
                detection_result.get('top_predictions', [])
            ),
            'next_steps': ExplanationGenerator._get_next_steps(
                is_disease, severity, crop
            )
        }
    
    @staticmethod
    def explain_soil_assessment(assessment: Dict) -> Dict:
        """
        Generate explanation for soil suitability assessment
        
        Args:
            assessment: Output from SuitabilityEngine.assess_soil_suitability
        
        Returns:
            Explanation dictionary
        """
        if not assessment.get('success', False):
            return {
                'status': 'Error',
                'message': assessment.get('error', 'Assessment failed'),
                'explanation': None
            }
        
        crop = assessment.get('crop_display_name', 'Unknown crop')
        score = assessment.get('overall_suitability_score', 0)
        rating = assessment.get('overall_suitability_rating', 'Unknown')
        params = assessment.get('parameter_assessments', {})
        
        # Overall assessment
        overall_text = f"""
SOIL SUITABILITY ASSESSMENT FOR {crop.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Suitability Score: {score:.2f}/1.0 ({rating})
"""
        
        if score >= 0.8:
            overall_text += "✓ Excellent - Soil is well-suited for cultivation"
        elif score >= 0.6:
            overall_text += "✓ Good - Soil is suitable with minor adjustments"
        elif score >= 0.4:
            overall_text += "⚠️  Fair - Soil needs improvement before planting"
        else:
            overall_text += "✗ Poor - Significant soil amendments required"
        
        # Parameter breakdown
        param_text = "\nPARAMETER BREAKDOWN:\n" + "─" * 40 + "\n"
        
        for param_name, assessment_data in params.items():
            value = assessment_data.get('value')
            optimal = assessment_data.get('optimal_value')
            status = assessment_data.get('status', 'unknown')
            message = assessment_data.get('message', '')
            
            if value is not None:
                icon = "✓" if status == 'optimal' else ("△" if status in ['low', 'high'] else "?")
                param_text += f"\n{param_name.upper()}\n"
                param_text += f"  Current: {value:.2f} | Optimal: {optimal:.2f}\n"
                param_text += f"  Status: {status.upper()} - {message}\n"
        
        # Recommendations
        recs = assessment.get('recommendations', [])
        rec_text = "\nACTION ITEMS:\n" + "─" * 40 + "\n"
        for i, rec in enumerate(recs, 1):
            rec_text += f"{i}. {rec}\n"
        
        full_explanation = overall_text + param_text + rec_text
        
        return {
            'status': 'Success',
            'overall_assessment': overall_text,
            'parameter_details': param_text,
            'action_items': rec_text,
            'full_explanation': full_explanation,
            'recommendations': recs,
            'next_steps': [
                "1. Implement recommended soil amendments",
                "2. Retest soil after 2-4 weeks if amendments applied",
                "3. Monitor crop response during growing season",
                "4. Adjust management practices based on growth"
            ]
        }
    
    @staticmethod
    def explain_weather_risk(weather_assessment: Dict) -> Dict:
        """
        Generate explanation for weather-based disease risk
        
        Args:
            weather_assessment: Output from SuitabilityEngine.assess_weather_risk
        
        Returns:
            Explanation dictionary
        """
        if not weather_assessment.get('success', False):
            return {
                'status': 'Error',
                'message': weather_assessment.get('error', 'Assessment failed'),
                'explanation': None
            }
        
        crop = weather_assessment.get('crop', 'Unknown crop')
        weather_summary = weather_assessment.get('weather_summary', {})
        overall_risk = weather_assessment.get('overall_risk_level', 'Unknown')
        disease_risks = weather_assessment.get('disease_risks', [])
        
        # Weather summary
        weather_text = f"""
WEATHER-BASED DISEASE RISK ASSESSMENT FOR {crop.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEATHER CONDITIONS (30-day average):
  Temperature: {weather_summary.get('avg_temperature_c', '?'):.1f}°C
  Humidity: {weather_summary.get('avg_humidity_percent', '?'):.1f}%
  Rainfall: {weather_summary.get('total_rainfall_mm', '?'):.1f}mm (30 days)

OVERALL DISEASE RISK LEVEL: {overall_risk}
"""
        
        if overall_risk in ['HIGH', 'VERY_HIGH']:
            weather_text += "⚠️  ALERT: Current weather conditions favor disease development"
        elif overall_risk == 'MODERATE':
            weather_text += "△ Moderate risk - Increased vigilance recommended"
        else:
            weather_text += "✓ Low risk - Current conditions favorable for healthy crop"
        
        # Top disease risks
        risks_text = "\nTOP DISEASES AT RISK:\n" + "─" * 40 + "\n"
        for i, risk in enumerate(disease_risks[:5], 1):
            disease = risk.get('disease', 'Unknown')
            risk_level = risk.get('risk_level', 'Unknown')
            risk_text = "\n".join(risk.get('risk_factors', [])[:2])
            
            icon = "🔴" if risk_level == 'VERY_HIGH' else ("🟠" if risk_level == 'HIGH' else "🟡")
            risks_text += f"\n{i}. {disease.replace('_', ' ').title()} [{icon} {risk_level}]\n"
            risks_text += f"   {risk_text}\n"
        
        # Recommendations
        recs = weather_assessment.get('recommendations', [])
        rec_text = "\nRECOMMENDED ACTIONS:\n" + "─" * 40 + "\n"
        for rec in recs:
            rec_text += f"• {rec}\n"
        
        full_explanation = weather_text + risks_text + rec_text
        
        return {
            'status': 'Success',
            'weather_summary': weather_text,
            'disease_risks': risks_text,
            'recommendations': rec_text,
            'full_explanation': full_explanation,
            'action_priority': ExplanationGenerator._get_action_priority(overall_risk)
        }
    
    @staticmethod
    def _explain_confidence(confidence: float, all_predictions: List[Dict]) -> Dict:
        """Explain model confidence level"""
        if confidence >= 0.9:
            level = "Very High"
            interpretation = "Model is highly confident in this identification"
        elif confidence >= 0.75:
            level = "High"
            interpretation = "Model is reasonably confident in this identification"
        elif confidence >= 0.6:
            level = "Moderate"
            interpretation = "Consider this a probable identification; visual confirmation recommended"
        else:
            level = "Low"
            interpretation = "Result is uncertain; other diseases are possible"
        
        # Alternative predictions
        alternatives = ""
        if len(all_predictions) > 1:
            alternatives = "\nOther likely possibilities:\n"
            for pred in all_predictions[1:4]:  # Show top 3 alternatives
                alt_name = pred.get('disease_name', 'Unknown')
                alt_conf = pred.get('confidence_percent', 0)
                alternatives += f"  • {alt_name} ({alt_conf:.1f}%)\n"
        
        return {
            'confidence_level': level,
            'confidence_percentage': confidence,
            'interpretation': interpretation,
            'alternatives': alternatives,
            'recommendation': "If uncertain, consult with agricultural extension officers"
        }
    
    @staticmethod
    def _get_next_steps(is_disease: bool, severity: str, crop: str) -> List[str]:
        """Get next steps based on detection"""
        steps = []
        
        if is_disease:
            if severity == 'severe':
                steps.extend([
                    "1. IMMEDIATE: Apply treatment following local guidelines",
                    "2. Isolate affected area to prevent spread",
                    "3. Monitor treatment effectiveness (daily)",
                    "4. Consult with local agricultural extension officer"
                ])
            elif severity == 'moderate':
                steps.extend([
                    "1. Implement field sanitation (remove infected parts)",
                    "2. Prepare fungicide/bactericide treatment",
                    "3. Increase monitoring to 3x per week",
                    "4. Contact extension officer for treatment approval"
                ])
            else:
                steps.extend([
                    "1. Monitor plant closely for disease progression",
                    "2. Improve air circulation and cultural practices",
                    "3. Check treatment options locally",
                    "4. Document the condition with photos"
                ])
        else:
            steps.extend([
                "1. Continue regular monitoring practices",
                "2. Maintain optimal growing conditions",
                "3. Document plant health status periodically",
                "4. Prevent introduction of disease vectors"
            ])
        
        return steps
    
    @staticmethod
    def _get_action_priority(risk_level: str) -> str:
        """Get action priority based on risk level"""
        priority_map = {
            'VERY_HIGH': 'URGENT - Act immediately',
            'HIGH': 'HIGH - Act within 24-48 hours',
            'MODERATE': 'MEDIUM - Plan interventions within this week',
            'LOW': 'LOW - Continue monitoring, preventive measures',
            'VERY_LOW': 'MINIMAL - Standard management practices'
        }
        return priority_map.get(risk_level, 'Unknown priority')


if __name__ == '__main__':
    # Test explanation generator
    gen = ExplanationGenerator()
    print("Explanation generator ready")
