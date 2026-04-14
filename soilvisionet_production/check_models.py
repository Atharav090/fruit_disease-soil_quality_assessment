import sys, os
sys.path.insert(0, 'soilvisionet_production')
os.environ['TRANSFORMERS_OFFLINE']='1'
from core.inference_engine import InferenceEngine
eng = InferenceEngine(models_path='results', device=None)
print('Device:', eng.device)
print('Models loaded:', list(eng.models.keys()))
