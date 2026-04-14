"""
Suitability Engine
Assesses soil and weather suitability for crops and predicts disease risk
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuitabilityEngine:
    """Evaluate crop suitability and disease risk based on soil and weather data"""
    
    def __init__(self, crop_db_path: str = 'config/crop_database.json',
                 disease_db_path: str = 'config/disease_database.json'):
        """
        Initialize suitability engine
        
        Args:
            crop_db_path: Path to crop database JSON
            disease_db_path: Path to disease database JSON
        """
        self.crop_db = {}
        self.disease_db = {}
        
        self._load_databases(crop_db_path, disease_db_path)
    
    def _load_databases(self, crop_db_path: str, disease_db_path: str):
        """Load crop and disease databases"""
        try:
            with open(crop_db_path, 'r') as f:
                self.crop_db = json.load(f)
            logger.info(f"Loaded {len(self.crop_db)} crop records")
        except Exception as e:
            logger.error(f"Failed to load crop database: {e}")
        
        try:
            with open(disease_db_path, 'r') as f:
                self.disease_db = json.load(f)
            logger.info(f"Loaded {len(self.disease_db)} disease records")
        except Exception as e:
            logger.error(f"Failed to load disease database: {e}")
    
    def assess_soil_suitability(self, crop: str, soil_params: Dict) -> Dict:
        """
        Assess soil suitability for a crop
        
        Args:
            crop: Crop type (normalized name, e.g., 'tomato')
            soil_params: Dictionary with soil parameters
                - soil_nitrogen (N): mg/kg
                - soil_phosphorus (P): mg/kg
                - soil_potassium (K): mg/kg
                - soil_ph: pH value
        
        Returns:
            Suitability assessment dictionary
        """
        if crop not in self.crop_db:
            return {
                'success': False,
                'error': f'Crop "{crop}" not found in database',
                'available_crops': list(self.crop_db.keys())
            }
        
        crop_info = self.crop_db[crop]
        optimal_soil = crop_info.get('optimal_soil', {})
        
        # Assess each parameter
        assessments = {
            'nitrogen': self._assess_parameter(
                soil_params.get('soil_nitrogen'),
                optimal_soil.get('nitrogen', {})
            ),
            'phosphorus': self._assess_parameter(
                soil_params.get('soil_phosphorus'),
                optimal_soil.get('phosphorus', {})
            ),
            'potassium': self._assess_parameter(
                soil_params.get('soil_potassium'),
                optimal_soil.get('potassium', {})
            ),
            'ph': self._assess_parameter(
                soil_params.get('soil_ph'),
                optimal_soil.get('ph', {})
            )
        }
        
        # Calculate overall score
        scores = [a['score'] for a in assessments.values() if a['score'] is not None]
        overall_score = np.mean(scores) if scores else None
        
        return {
            'success': True,
            'crop': crop,
            'crop_display_name': crop_info.get('display_name', crop),
            'overall_suitability_score': overall_score,
            'overall_suitability_rating': self._score_to_rating(overall_score),
            'parameter_assessments': assessments,
            'input_soil_params': soil_params,
            'recommended_soil_params': optimal_soil,
            'recommendations': self._get_soil_recommendations(assessments, crop)
        }
    
    def assess_weather_risk(self, crop: str, weather_sequence: List[Dict]) -> Dict:
        """
        Assess disease risk based on weather conditions
        
        Args:
            crop: Crop type
            weather_sequence: List of 30 daily weather records
                Each with keys: 'temp' (°C), 'rainfall' (mm), 'humidity' (%)
        
        Returns:
            Disease risk assessment dictionary
        """
        if not weather_sequence or len(weather_sequence) == 0:
            return {
                'success': False,
                'error': 'Empty weather sequence'
            }
        
        # Extract averages from weather sequence
        temps = [w.get('temp', 0) for w in weather_sequence]
        rainfall = [w.get('rainfall', 0) for w in weather_sequence]
        humidity = [w.get('humidity', 0) for w in weather_sequence]
        
        avg_temp = np.mean(temps)
        avg_rainfall = np.mean(rainfall)
        avg_humidity = np.mean(humidity)
        total_rainfall = np.sum(rainfall)
        
        # Find diseases for this crop
        crop_diseases = []
        for disease_name, disease_info in self.disease_db.items():
            if disease_info.get('crop') == crop:
                crop_diseases.append((disease_name, disease_info))
        
        if not crop_diseases:
            return {
                'success': False,
                'error': f'No diseases found for crop "{crop}"'
            }
        
        # Calculate risk for each disease
        disease_risks = []
        for disease_name, disease_info in crop_diseases:
            risk_score, risk_factors = self._calculate_disease_risk(
                disease_name, disease_info, avg_temp, avg_rainfall, avg_humidity
            )
            
            disease_risks.append({
                'disease': disease_name,
                'risk_score': risk_score,
                'risk_level': self._score_to_risk_level(risk_score),
                'risk_factors': risk_factors
            })
        
        # Sort by risk score (highest first)
        disease_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {
            'success': True,
            'crop': crop,
            'weather_summary': {
                'avg_temperature_c': round(avg_temp, 2),
                'avg_humidity_percent': round(avg_humidity, 2),
                'total_rainfall_mm': round(total_rainfall, 2),
                'period_days': len(weather_sequence)
            },
            'disease_risks': disease_risks,
            'highest_risk_disease': disease_risks[0] if disease_risks else None,
            'overall_risk_level': self._score_to_risk_level(
                disease_risks[0]['risk_score'] if disease_risks else 0
            ),
            'recommendations': self._get_weather_recommendations(disease_risks[:3], crop)
        }
    
    def _assess_parameter(self, current_value: Optional[float], 
                         optimal_range: Dict) -> Dict:
        """
        Assess a single soil parameter
        
        Args:
            current_value: Current measured value
            optimal_range: Dict with 'min', 'max', 'optimal'
        
        Returns:
            Assessment dictionary
        """
        if current_value is None:
            return {
                'value': None,
                'status': 'unknown',
                'score': None,
                'message': 'No data available'
            }
        
        optimal = optimal_range.get('optimal', (optimal_range.get('min', 0) +
                                               optimal_range.get('max', 100)) / 2)
        min_val = optimal_range.get('min', 0)
        max_val = optimal_range.get('max', 100)
        
        # Calculate distance from optimal
        if current_value < min_val:
            status = 'low'
            distance = min_val - current_value
            score = max(0, 1 - (distance / min_val)) if min_val > 0 else 0.5
        elif current_value > max_val:
            status = 'high'
            distance = current_value - max_val
            score = max(0, 1 - (distance / max_val))
        else:
            status = 'optimal'
            # Score based on closeness to optimal
            distance = abs(current_value - optimal)
            max_distance = max(optimal - min_val, max_val - optimal)
            score = 1 - (distance / max_distance) if max_distance > 0 else 1.0
        
        return {
            'value': current_value,
            'optimal_value': optimal,
            'optimal_range': [min_val, max_val],
            'status': status,
            'score': min(1.0, max(0.0, score)),
            'message': self._param_status_message(status, current_value, optimal_range)
        }
    
    def _calculate_disease_risk(self, disease_name: str, disease_info: Dict,
                               temp: float, rainfall: float, 
                               humidity: float) -> Tuple[float, List[str]]:
        """
        Calculate risk score for a specific disease
        
        Args:
            disease_name: Disease name
            disease_info: Disease database entry
            temp: Average temperature (°C)
            rainfall: Average daily rainfall (mm)
            humidity: Average humidity (%)
        
        Returns:
            (risk_score [0-1], list of risk factors)
        """
        risk_score = 0.0
        risk_factors = []
        num_factors = 0
        
        weather_req = disease_info.get('weather_requirements', {})
        
        # Temperature check
        temp_range = weather_req.get('temperature', {})
        if temp_range:
            temp_optimal = temp_range.get('optimal', 22)
            temp_min = temp_range.get('min', 15)
            temp_max = temp_range.get('max', 30)
            
            if temp_min <= temp <= temp_max:
                # Within range
                temp_distance = abs(temp - temp_optimal)
                max_distance = max(temp_optimal - temp_min, temp_max - temp_optimal)
                temp_score = 1 - (temp_distance / max_distance) if max_distance > 0 else 1
            else:
                temp_score = 0
            
            risk_score += temp_score
            num_factors += 1
            
            if temp_score > 0.7:
                risk_factors.append(f"Temperature ({temp}°C) favors disease development")
        
        # Humidity check
        humidity_range = weather_req.get('humidity', {})
        if humidity_range:
            humidity_optimal = humidity_range.get('optimal', 70)
            humidity_min = humidity_range.get('min', 40)
            humidity_max = humidity_range.get('max', 90)
            
            if humidity_min <= humidity <= humidity_max:
                humidity_distance = abs(humidity - humidity_optimal)
                max_distance = max(humidity_optimal - humidity_min, 
                                 humidity_max - humidity_optimal)
                humidity_score = 1 - (humidity_distance / max_distance) if max_distance > 0 else 1
            else:
                humidity_score = 0
            
            risk_score += humidity_score
            num_factors += 1
            
            if humidity_score > 0.7:
                risk_factors.append(f"High humidity ({humidity}%) creates favorable conditions")
        
        # Rainfall check
        rainfall_range = weather_req.get('rainfall', {})
        if rainfall_range:
            rainfall_optimal = rainfall_range.get('optimal', 3)
            rainfall_min = rainfall_range.get('min', 0)
            rainfall_max = rainfall_range.get('max', 10)
            
            if rainfall_min <= rainfall <= rainfall_max:
                rainfall_distance = abs(rainfall - rainfall_optimal)
                max_distance = max(rainfall_optimal - rainfall_min,
                                 rainfall_max - rainfall_optimal)
                rainfall_score = 1 - (rainfall_distance / max_distance) if max_distance > 0 else 1
            else:
                rainfall_score = 0
            
            risk_score += rainfall_score
            num_factors += 1
            
            if rainfall_score > 0.7:
                risk_factors.append(f"Rainfall patterns ({rainfall}mm avg) suit disease spread")
        
        # Normalize score
        risk_score = risk_score / num_factors if num_factors > 0 else 0
        
        if not risk_factors:
            risk_factors.append("No specific weather factors favor this disease")
        
        return risk_score, risk_factors
    
    def assess_crop_suitability_comprehensive(self, crop: str, 
                                              soil_params: Dict,
                                              weather_sequence: List[Dict]) -> Dict:
        """
        Comprehensive assessment combining soil and weather
        
        Args:
            crop: Crop type
            soil_params: Soil parameters
            weather_sequence: Weather sequence
        
        Returns:
            Combined assessment
        """
        soil_assessment = self.assess_soil_suitability(crop, soil_params)
        weather_assessment = self.assess_weather_risk(crop, weather_sequence)
        
        return {
            'crop': crop,
            'soil_assessment': soil_assessment,
            'weather_assessment': weather_assessment,
            'final_recommendation': self._get_final_recommendation(
                soil_assessment, weather_assessment, crop
            )
        }
    
    def _get_soil_recommendations(self, assessments: Dict, crop: str) -> List[str]:
        """Generate soil management recommendations"""
        recommendations = []
        
        nitrogen = assessments.get('nitrogen', {})
        if nitrogen.get('status') == 'low':
            recommendations.append("Apply nitrogen-rich fertilizer (compost, manure, or urea)")
        elif nitrogen.get('status') == 'high':
            recommendations.append("Reduce nitrogen inputs; consider removing organic matter")
        
        phosphorus = assessments.get('phosphorus', {})
        if phosphorus.get('status') == 'low':
            recommendations.append("Add phosphorus fertilizer (bone meal, phosphate rock)")
        elif phosphorus.get('status') == 'high':
            recommendations.append("Avoid phosphorus-rich fertilizers for 1-2 seasons")
        
        potassium = assessments.get('potassium', {})
        if potassium.get('status') == 'low':
            recommendations.append("Apply potassium fertilizer (potash, wood ash) or compost")
        elif potassium.get('status') == 'high':
            recommendations.append("Reduce potassium amendments; focus on other nutrients")
        
        ph = assessments.get('ph', {})
        if ph.get('status') == 'low':
            recommendations.append("Raise pH using lime; test soil pH regularly")
        elif ph.get('status') == 'high':
            recommendations.append("Lower pH by adding sulfur or acidifying agents")
        
        if not recommendations:
            recommendations.append("Soil conditions are suitable for crop cultivation")
        
        return recommendations
    
    def _get_weather_recommendations(self, top_risks: List[Dict], crop: str) -> List[str]:
        """Generate weather-based disease management recommendations"""
        recommendations = []
        
        if not top_risks:
            recommendations.append("Current weather conditions are low risk for major diseases")
            return recommendations
        
        high_risk_diseases = [r for r in top_risks if r['risk_level'] in ['HIGH', 'VERY_HIGH']]
        
        if high_risk_diseases:
            recommendations.append(f"⚠️ High disease risk detected: {high_risk_diseases[0]['disease']}")
            recommendations.append("Increase monitoring frequency to every 2-3 days")
            recommendations.append("Prepare fungicide treatments (consult local extension officer)")
            recommendations.append("Improve air circulation; reduce humidity where possible")
        
        for risk in top_risks[:2]:
            for factor in risk.get('risk_factors', [])[:1]:  # First factor only
                recommendations.append(f"{factor}")
        
        return recommendations
    
    def _get_final_recommendation(self, soil_assess: Dict, weather_assess: Dict,
                                 crop: str) -> Dict:
        """Generate final planting/management recommendation"""
        soil_score = soil_assess.get('overall_suitability_score', 0)
        weather_risk = weather_assess.get('overall_risk_level', 'MODERATE')
        
        # Map weather risk to numeric score (inverted)
        weather_score_map = {
            'VERY_LOW': 1.0,
            'LOW': 0.8,
            'MODERATE': 0.6,
            'HIGH': 0.3,
            'VERY_HIGH': 0.1
        }
        weather_score = weather_score_map.get(weather_risk, 0.5)
        
        # Combined score
        combined_score = (soil_score * 0.4 + weather_score * 0.6) if soil_score else weather_score
        
        # Recommendation
        if combined_score >= 0.8:
            recommendation = "✓ EXCELLENT conditions - Plant now with standard management"
        elif combined_score >= 0.6:
            recommendation = "✓ GOOD conditions - Plant with moderate monitoring"
        elif combined_score >= 0.4:
            recommendation = "⚠️  FAIR conditions - Plant with extra care and monitoring"
        else:
            recommendation = "✗ POOR conditions - Delay planting or apply remedial measures"
        
        return {
            'combined_score': combined_score,
            'soil_score': soil_score,
            'weather_risk_level': weather_risk,
            'recommendation_text': recommendation,
            'action_items': self._get_action_items(soil_assess, weather_assess, crop)
        }
    
    def _get_action_items(self, soil_assess: Dict, weather_assess: Dict, crop: str) -> List[str]:
        """Generate action items"""
        actions = []
        
        # Soil-related actions
        if soil_assess.get('success'):
            soil_recs = soil_assess.get('recommendations', [])
            actions.extend(soil_recs[:2])
        
        # Weather-related actions
        if weather_assess.get('success'):
            weather_recs = weather_assess.get('recommendations', [])
            actions.extend(weather_recs[:2])
        
        # Add general practices
        actions.append(f"Maintain detailed records for {crop} cultivation")
        actions.append("Schedule regular monitoring (at least 2x per week)")
        
        return actions
    
    @staticmethod
    def _score_to_rating(score: Optional[float]) -> str:
        """Convert score to rating"""
        if score is None:
            return 'UNKNOWN'
        elif score >= 0.8:
            return 'EXCELLENT'
        elif score >= 0.6:
            return 'GOOD'
        elif score >= 0.4:
            return 'FAIR'
        else:
            return 'POOR'
    
    @staticmethod
    def _score_to_risk_level(score: float) -> str:
        """Convert risk score to level"""
        if score >= 0.8:
            return 'VERY_HIGH'
        elif score >= 0.6:
            return 'HIGH'
        elif score >= 0.4:
            return 'MODERATE'
        elif score >= 0.2:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    @staticmethod
    def _param_status_message(status: str, current: float, optimal: Dict) -> str:
        """Generate status message for parameter"""
        if status == 'optimal':
            return "Within optimal range"
        elif status == 'low':
            return f"Below optimal ({current:.1f} < {optimal.get('optimal', '?'):.1f})"
        else:
            return f"Above optimal ({current:.1f} > {optimal.get('optimal', '?'):.1f})"


if __name__ == '__main__':
    # Test suitability engine
    engine = SuitabilityEngine()
    
    # Test data
    soil_params = {
        'soil_nitrogen': 75,
        'soil_phosphorus': 40,
        'soil_potassium': 30,
        'soil_ph': 6.8
    }
    
    weather = [
        {'temp': 25, 'rainfall': 2, 'humidity': 65} for _ in range(30)
    ]
    
    result = engine.assess_crop_suitability_comprehensive('tomato', soil_params, weather)
    print("Test complete")
