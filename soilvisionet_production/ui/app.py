"""
SoilVisioNet Production UI
Streamlit application for disease detection and soil/weather assessment
"""

import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.inference_engine import InferenceEngine
from core.image_processor import ImageProcessor
from modules.disease_detector import DiseaseDetector
from modules.suitability_engine import SuitabilityEngine
from modules.explanation_generator import ExplanationGenerator
from modules.soil_cause_analyzer import SoilCauseAnalyzer


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def render_soil_analysis_ui(result: dict):
    """
    Render soil analysis UI sections based on integrated soil data.
    
    Shows simulated sensor readings, contributing factors, and healthy vs current comparison.
    """
    soil_analysis = result.get('soil_factor_analysis')
    
    if not soil_analysis or not soil_analysis.get('success'):
        return
    
    st.divider()
    st.subheader("🌱 Soil Factor Analysis")
    
    # Section A: Simulated Sensor Readings
    st.markdown("**📡 Simulated Sensor Readings**")
    st.caption("Current soil parameter readings from simulated sensors")
    
    readings = soil_analysis.get('current_readings', {})
    if readings:
        cols = st.columns(4)
        param_display = {
            'nitrogen': ('Nitrogen', 'ppm'),
            'phosphorus': ('Phosphorus', 'ppm'), 
            'potassium': ('Potassium', 'ppm'),
            'ph': ('Soil pH', 'pH')
        }
        
        for i, (param, (display_name, unit)) in enumerate(param_display.items()):
            with cols[i]:
                value = readings.get(param, {}).get('value', 'N/A')
                st.metric(display_name, f"{value} {unit}")
    else:
        st.info("No sensor readings available")
    
    st.markdown("---")
    
    # Section B: Likely Contributing Soil Factors
    st.markdown("**🔍 Likely Contributing Soil Factors**")
    st.caption("Soil conditions that may contribute to disease development")
    
    factors = soil_analysis.get('contributing_factors', [])
    if factors:
        for factor in factors:
            st.markdown(f"• {factor}")
    else:
        st.info("No specific soil factors identified")
    
    st.markdown("---")
    
    # Section C: Healthy vs Current Comparison
    st.markdown("**⚖️ Healthy vs Current Comparison**")
    st.caption("Compare current readings against healthy baseline ranges")
    
    if readings:
        # Create a nice table-like display
        comparison_data = []
        
        for param in ['nitrogen', 'phosphorus', 'potassium', 'ph']:
            reading = readings.get(param, {})
            if reading:
                current_val = reading.get('value', 'N/A')
                status = reading.get('status', 'unknown')
                healthy_range = reading.get('healthy_range', {})
                min_val = healthy_range.get('min', 'N/A')
                max_val = healthy_range.get('max', 'N/A')
                unit = healthy_range.get('unit', '')
                
                # Status badge
                if status == 'optimal':
                    status_badge = "🟢 Optimal"
                elif status == 'low':
                    status_badge = "🔴 Low"
                elif status == 'high':
                    status_badge = "🟠 High"
                else:
                    status_badge = "⚪ Unknown"
                
                comparison_data.append({
                    'Parameter': param.title(),
                    'Current': f"{current_val} {unit}",
                    'Healthy Range': f"{min_val}-{max_val} {unit}",
                    'Status': status_badge
                })
        
        if comparison_data:
            st.dataframe(comparison_data, use_container_width=True)
        else:
            st.info("Comparison data not available")
    
    # Disclaimer
    st.caption("⚠️ **Disclaimer:** These are simulated sensor readings for demonstration purposes. Actual soil testing is recommended for accurate results.")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SoilVisioNet",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 SoilVisioNet - Intelligent Fruit Disease & Soil Management System")
st.markdown("""
**AI-powered disease detection + soil suitability assessment for sustainable farming**

Analyze fruit/leaf images to detect diseases automatically and assess soil conditions for optimal crop cultivation.
""")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize session state
    if 'detector_initialized' not in st.session_state:
        st.session_state.detector_initialized = False
    if 'suitability_initialized' not in st.session_state:
        st.session_state.suitability_initialized = False
    if 'soil_analyzer_initialized' not in st.session_state:
        st.session_state.soil_analyzer_initialized = False
    
    # Initialize models
    with st.spinner("Initializing AI models..."):
        try:
            if not st.session_state.detector_initialized:
                st.session_state.disease_detector = DiseaseDetector(
                    models_path='../results/hybrid',
                    device='auto'  # Auto-detect best device (CUDA with fallback to CPU)
                )
                st.session_state.detector_initialized = True
                st.success("✓ Disease detection models loaded")
            
            if not st.session_state.suitability_initialized:
                st.session_state.suitability_engine = SuitabilityEngine(
                    crop_db_path='config/crop_database.json',
                    disease_db_path='config/disease_database.json'
                )
                st.session_state.suitability_initialized = True
                st.success("✓ Suitability assessment models loaded")
                
            if not st.session_state.soil_analyzer_initialized:
                st.session_state.soil_cause_analyzer = SoilCauseAnalyzer()
                st.session_state.soil_analyzer_initialized = True
                st.success("✓ Soil cause analyzer loaded")
                
            st.session_state.explanation_gen = ExplanationGenerator()
        except Exception as e:
            st.error(f"⚠️ Initialization error: {str(e)}")
            st.info("Some features may be limited. Check file paths if models are missing.")
    
    st.divider()
    
    # Fixed model (SoilVisioNet Hybrid)
    st.subheader("Model Settings")
    st.info("🤖 **Model**: SoilVisioNet (Advanced Hybrid Architecture)\n\nCombines Vision Transformer + ELM for optimal disease detection accuracy")
    st.session_state.selected_model = 'hybrid'
    
    # Info
    st.divider()
    st.subheader("ℹ️ About")
    st.info("""
    **SoilVisioNet v1.0**
    
    • Disease Detection: SoilVisioNet Hybrid Model
    • Accuracy: 98%+ on 55 disease classes
    • Soil Assessment: 4 key parameters
    • Weather Risk: Real-time prediction
    
    **Crops**: Tomato, Potato, Apple, Mango, Grape, and 12+ more
    """)


# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2 = st.tabs(["🔬 Disease Detection", "🌱 Soil & Weather Assessment"])

# ============================================================================
# TAB 1: DISEASE DETECTION
# ============================================================================

with tab1:
    st.header("🔬 Fruit/Leaf Disease Detection")
    st.markdown("Upload an image of a fruit, leaf, or plant part for instant disease analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Get image info
            image_array = np.array(image)
            stats = ImageProcessor.get_image_stats(image_array)
            st.caption(f"Image: {stats['shape'][0]}×{stats['shape'][1]}px")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🧠 Analysis")
            
            with st.spinner("Running disease detection model..."):
                try:
                    # Run detection
                    result = st.session_state.disease_detector.detect_from_array(
                        image_array,
                        use_model=st.session_state.selected_model,
                        return_top_n=5
                    )
                    
                    # Run soil cause analysis
                    # Run soil cause analysis safely
                    soil_analysis = None
                    simulated_sensor_readings = {}

                    if result.get('success', False) and 'primary_prediction' in result:
                        try:
                            analysis = st.session_state.soil_cause_analyzer.analyze_prediction(
                                result['primary_prediction']
                            )

                            if analysis.get('success'):
                                soil_analysis = analysis
                                simulated_sensor_readings = analysis.get('current_readings', {})
                            else:
                                soil_analysis = None
                                simulated_sensor_readings = {}

                        except Exception as soil_err:
                            soil_analysis = None
                            simulated_sensor_readings = {}
                            st.warning(f"Soil analysis unavailable for this prediction: {soil_err}")

                    result['soil_factor_analysis'] = soil_analysis
                    result['simulated_sensor_readings'] = simulated_sensor_readings
                    
                    if result.get('success', False):
                        # Primary prediction
                        pred = result.get('primary_prediction', {})
                        
                        # Large confidence metric
                        col_conf1, col_conf2, col_conf3 = st.columns(3)
                        with col_conf1:
                            st.metric(
                                "Confidence",
                                f"{pred.get('confidence_percent', 0):.1f}%"
                            )
                        with col_conf2:
                            severity = pred.get('severity', 'Unknown').upper()
                            color = "🔴" if severity == 'SEVERE' else ("🟠" if severity == 'MODERATE' else "🟡")
                            st.metric("Severity", f"{color} {severity}")
                        with col_conf3:
                            is_healthy = "Healthy" if not pred.get('is_disease') else "Disease"
                            st.metric("Status", is_healthy)
                        
                        st.divider()
                        
                        # Prediction details
                        st.subheader("📋 Detected Disease")
                        disease_name = pred.get('disease_name', 'Unknown')
                        
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.metric("Disease", disease_name.replace('_', ' ').title())
                            st.metric("Fruit", pred.get('crop', 'Unknown').title())
                        with col_right:
                            st.metric("Condition", pred.get('condition', 'Unknown').title())
                            st.metric("Model Used", "SoilVisioNet")
                        
                        # Render soil analysis UI if available
                        render_soil_analysis_ui(result)
                        
                    else:
                        st.error(f"Detection failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    st.info("Try with a different image or check model files")
    
    # Detailed results section
    if uploaded_file is not None and 'result' in locals() and result.get('success'):
        st.divider()
        st.subheader("📊 Detailed Results")
        
        # Top predictions table
        st.markdown("**Top 5 Predictions:**")
        top_preds = result.get('top_predictions', [])
        
        pred_data = []
        for i, pred in enumerate(top_preds, 1):
            pred_data.append({
                'Rank': i,
                'Disease': pred.get('disease_name', '').replace('_', ' ').title(),
                'Confidence': f"{pred.get('confidence_percent', 0):.1f}%",
                'Severity': pred.get('severity', 'Unknown').title(),
                'fruit': pred.get('crop', 'Unknown').title()
            })
        
        st.dataframe(pred_data, use_container_width=True)
        
        st.divider()
        
        # Explanation and recommendations
        explanation = st.session_state.explanation_gen.explain_detection(result)
        
        st.subheader("💡 Analysis & Recommendations")
        
        if explanation.get('status') == 'Success':
            # Expandable sections
            with st.expander("📖 Detailed Explanation", expanded=True):
                st.markdown(explanation.get('main_explanation', ''))
            
            if explanation.get('symptoms_explanation'):
                with st.expander("🔍 Symptoms Details"):
                    st.markdown(explanation.get('symptoms_explanation', ''))
            
            if explanation.get('treatment_explanation'):
                with st.expander("💊 Treatment Options"):
                    st.markdown(explanation.get('treatment_explanation', ''))
            
            # Confidence analysis
            with st.expander("📊 Model Confidence Analysis"):
                conf_analysis = explanation.get('confidence_analysis', {})
                st.markdown(f"""
                **Confidence Level:** {conf_analysis.get('confidence_level')}
                
                {conf_analysis.get('interpretation', '')}
                
                {conf_analysis.get('alternatives', '')}
                """)
            
            # Next steps
            st.info("**Next Steps:**")
            for step in explanation.get('next_steps', []):
                st.markdown(f"  • {step}")
        
        # Export results button
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("📥 Download Results as JSON"):
                json_str = json.dumps(result, indent=2, default=str)
                st.download_button(
                    "Download JSON",
                    json_str,
                    "detection_result.json",
                    "application/json"
                )
        
        with col_export2:
            if st.button("📄 Download Report as Text"):
                report = f"""
SoilVisioNet Disease Detection Report
{'='*50}

PRIMARY PREDICTION:
{json.dumps(result.get('primary_prediction'), indent=2, default=str)}

EXPLANATION:
{explanation.get('full_explanation', 'N/A')}

RECOMMENDATIONS:
{chr(10).join(explanation.get('next_steps', []))}
"""
                st.download_button(
                    "Download Report",
                    report,
                    "detection_report.txt",
                    "text/plain"
                )


# ============================================================================
# TAB 2: SOIL & WEATHER ASSESSMENT
# ============================================================================

with tab2:
    st.header("🌱 Soil & Weather Suitability Assessment")
    st.markdown("Assess soil conditions and weather-based disease risk for your fruitss")
    
    # Input section
    st.subheader("📝 Input fruit and Soil Data")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.markdown("**Fruit Selection:**")
        
        # Get available crops
        try:
            crops_list = sorted(st.session_state.suitability_engine.crop_db.keys())
        except:
            crops_list = ['tomato', 'potato', 'apple', 'mango', 'grape']
        
        selected_crop = st.selectbox(
            "Select fruit:",
            crops_list,
            help="Choose the fruit you want to assess"
        )
        
        st.markdown("**Soil Parameters:**")
        soil_n = st.number_input(
            "Nitrogen (N) - mg/kg",
            min_value=0, max_value=300, value=75, step=5,
            help="Soil nitrogen content in mg/kg"
        )
        soil_p = st.number_input(
            "Phosphorus (P) - mg/kg",
            min_value=0, max_value=200, value=40, step=2,
            help="Soil phosphorus content in mg/kg"
        )
    
    with col_input2:
        soil_k = st.number_input(
            "Potassium (K) - mg/kg",
            min_value=0, max_value=300, value=30, step=2,
            help="Soil potassium content in mg/kg"
        )
        soil_ph = st.number_input(
            "Soil pH",
            min_value=3.0, max_value=9.0, value=6.8, step=0.1,
            help="Soil pH value (3-9)"
        )
    
    # Weather data input
    st.markdown("**Weather Data (30-day average):**")
    
    col_weather1, col_weather2, col_weather3 = st.columns(3)
    
    with col_weather1:
        avg_temp = st.number_input(
            "Avg Temperature (°C)",
            min_value=-20, max_value=50, value=25, step=1
        )
    
    with col_weather2:
        avg_rainfall = st.number_input(
            "Avg Daily Rainfall (mm)",
            min_value=0.0, max_value=50.0, value=2.0, step=0.5
        )
    
    with col_weather3:
        avg_humidity = st.number_input(
            "Avg Humidity (%)",
            min_value=0, max_value=100, value=65, step=1
        )
    
    # Analysis button
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        run_assessment = st.button("🔍 Analyze Suitability", type="primary")
    
    with col_btn2:
        use_sample_data = st.checkbox("Use example data")
    
    with col_btn3:
        st.write("")  # Spacer
    
    # Use sample data if requested
    if use_sample_data:
        soil_n, soil_p, soil_k = 75, 40, 30
        soil_ph = 6.8
        avg_temp = 25
        avg_rainfall = 2
        avg_humidity = 65
        st.info("✓ Example data loaded for optimal growing conditions")
    
    # Run assessment
    if run_assessment:
        try:
            soil_params = {
                'soil_nitrogen': soil_n,
                'soil_phosphorus': soil_p,
                'soil_potassium': soil_k,
                'soil_ph': soil_ph
            }
            
            # Create weather sequence (30 days with same values)
            weather_sequence = [
                {
                    'temp': avg_temp,
                    'rainfall': avg_rainfall,
                    'humidity': avg_humidity
                }
                for _ in range(30)
            ]
            
            # Run comprehensive assessment
            with st.spinner("Running comprehensive assessment..."):
                assessment = st.session_state.suitability_engine.assess_crop_suitability_comprehensive(
                    selected_crop,
                    soil_params,
                    weather_sequence
                )
            
            # Display results
            st.divider()
            st.subheader("📊 Assessment Results")
            
            # Final recommendation
            final_rec = assessment.get('final_recommendation', {})
            score = final_rec.get('combined_score', 0)
            rec_text = final_rec.get('recommendation_text', '')
            
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.metric(
                    "Overall Suitability Score",
                    f"{score:.2f}/1.0",
                    help="Combined soil and weather suitability"
                )
            with col_rec2:
                rating = "Excellent" if score >= 0.8 else ("Good" if score >= 0.6 else ("Fair" if score >= 0.4 else "Poor"))
                st.metric("Rating", rating)
            
            st.success(f"✨ {rec_text}")
            
            # Soil assessment
            st.divider()
            st.subheader("🌐 Soil Suitability Details")
            
            soil_assess = assessment.get('soil_assessment', {})
            soil_explanations = st.session_state.explanation_gen.explain_soil_assessment(soil_assess)
            
            with st.expander("Soil Parameters Breakdown", expanded=True):
                st.markdown(soil_explanations.get('parameter_details', ''))
            
            # Weather assessment
            st.divider()
            st.subheader("☁️ Weather-Based Disease Risk")
            
            weather_assess = assessment.get('weather_assessment', {})
            weather_explanations = st.session_state.explanation_gen.explain_weather_risk(weather_assess)
            
            col_weather_info1, col_weather_info2 = st.columns(2)
            with col_weather_info1:
                st.metric("Overall Disease Risk", weather_assess.get('overall_risk_level', 'Unknown'))
            with col_weather_info2:
                st.metric("Action Priority", weather_explanations.get('action_priority', 'Unknown'))
            
            with st.expander("Disease Risk Details", expanded=True):
                st.markdown(weather_explanations.get('disease_risks', ''))
            
            # Recommendations
            st.divider()
            st.subheader("✅ Recommended Actions")
            
            action_items = final_rec.get('action_items', [])
            for i, action in enumerate(action_items, 1):
                st.markdown(f"{i}. {action}")
            
            # Export
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("📥 Download Assessment as JSON"):
                    json_str = json.dumps(assessment, indent=2, default=str)
                    st.download_button(
                        "Download Results",
                        json_str,
                        f"{selected_crop}_assessment.json",
                        "application/json"
                    )
            
            with col_exp2:
                if st.button("📄 Download Management Plan"):
                    plan = f"""
SOIL & WEATHER MANAGEMENT PLAN
{'='*60}

Fruit: {selected_crop.title()}
Date: {st.session_state.get('assessment_date', 'N/A')}

EXECUTIVE SUMMARY:
{rec_text}

Suitability Score: {score:.2f}/1.0
{rating} conditions for {selected_crop.title()}

SOIL CONDITIONS:
{soil_explanations.get('parameter_details', 'N/A')}

WEATHER RISK ASSESSMENT:
{weather_explanations.get('disease_risks', 'N/A')}

CRITICAL ACTIONS REQUIRED:
{chr(10).join(action_items)}

NEXT STEPS:
{chr(10).join(weather_explanations.get('action_priority', []))}

---
Generated by SoilVisioNet v1.0
"""
                    st.download_button(
                        "Download Plan",
                        plan,
                        f"{selected_crop}_management_plan.txt",
                        "text/plain"
                    )
        
        except Exception as e:
            st.error(f"Error during assessment: {str(e)}")
            st.info("Check that fruit exists in database and all parameters are valid")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**SoilVisioNet v1.0** | Intelligent Fruit Disease Detection and Soil Quality Assessment System

*Powered by Vision Transformers, LSTM, and ELM deep learning models*
*Built with real agricultural data from 15,288 samples across 55 disease classes*

For support and documentation, refer to project documentation.
""")
