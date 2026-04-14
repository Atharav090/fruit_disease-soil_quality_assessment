"""
SoilVisioNet Evaluation Script
Computes F1, Precision, Recall, Accuracy, and Inference Time
on the test set using the pretrained hybrid fusion model.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

from data_loader import get_dataloaders
from train_elm_phase2b import SoilVisioNetELM
from train_lstm_phase2a import WeatherLSTM
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class FusionHead(nn.Module):
    """Fusion head matching training architecture."""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(FusionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class SoilVisioNet(nn.Module):
    """SoilVisioNet: Hybrid fusion of ViT + LSTM + ELM for disease detection."""
    def __init__(self, vit_model, lstm_model, elm_model, fusion_head):
        super(SoilVisioNet, self).__init__()
        self.vit = vit_model
        self.lstm = lstm_model
        self.elm = elm_model
        self.fusion = fusion_head

    def forward(self, images, weather, soil):
        # ViT logits and features
        vit_outputs = self.vit(pixel_values=images, output_hidden_states=True)
        vit_logits = vit_outputs.logits
        vit_features = vit_outputs.hidden_states[-1][:, 0, :]

        # ELM logits
        elm_logits = self.elm(vit_features, soil)

        # LSTM output
        lstm_out = self.lstm(weather, soil)
        if lstm_out.dim() == 1:
            lstm_logits = lstm_out.unsqueeze(1).float()
        else:
            lstm_logits = lstm_out.float()

        # Concatenate and fuse
        x = torch.cat([vit_logits, elm_logits, lstm_logits], dim=1)
        return self.fusion(x)


def main():
    config = {
        'data_root': 'data/unified_dataset',
        'batch_size': 16,
        'num_workers': 0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'vit_checkpoint': 'results/vit_phase1/best_model.pt',
        'lstm_checkpoint': 'results/lstm_phase2a/best_lstm_model.pt',
        'elm_checkpoint': 'results/elm_phase2b/elm_model.pt',
        'fusion_checkpoint': 'results/hybrid/best_hybrid_fusion.pt'
    }
    
    # Auto-detect GPU
    if torch.cuda.is_available():
        config['num_workers'] = 2
        print(f"[GPU DETECTED] {torch.cuda.get_device_name(0)}")
    else:
        config['batch_size'] = 8
        print(f"[CPU MODE] Evaluation on CPU")

    device = torch.device(config['device'])

    print("=" * 70)
    print("SoilVisioNet - COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # Load dataset
    print("\nLoading test dataset...")
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    num_classes = datasets['train'].num_classes

    # Load models
    print("\nLoading SoilVisioNet components...")
    
    # ViT
    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    if Path(config['vit_checkpoint']).exists():
        ck = torch.load(config['vit_checkpoint'], map_location=device)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            vit_model.load_state_dict(ck['model_state_dict'])
        else:
            try:
                vit_model.load_state_dict(ck)
            except Exception:
                print('Warning: could not load ViT checkpoint; using pretrained ViT')
    vit_model.to(config['device'])
    vit_model.eval()
    for p in vit_model.parameters():
        p.requires_grad = False

    # LSTM
    lstm_model = WeatherLSTM(num_classes=num_classes, hidden_size=256, num_layers=2, dropout=0.3)
    if Path(config['lstm_checkpoint']).exists():
        ck = torch.load(config['lstm_checkpoint'], map_location=device)
        try:
            lstm_model.load_state_dict(ck)
        except Exception:
            if isinstance(ck, dict) and 'model_state_dict' in ck:
                lstm_model.load_state_dict(ck['model_state_dict'])
    lstm_model.to(config['device'])
    lstm_model.eval()
    for p in lstm_model.parameters():
        p.requires_grad = False

    # ELM
    elm_model = SoilVisioNetELM(vit_feature_dim=768, soil_dim=4, hidden_size=512, num_classes=num_classes)
    if Path(config['elm_checkpoint']).exists():
        try:
            elm_state = torch.load(config['elm_checkpoint'], map_location=device)
            elm_model.load_state_dict(elm_state)
        except Exception:
            print('Warning: could not load ELM checkpoint')
    elm_model.to(config['device'])
    elm_model.eval()
    for p in elm_model.parameters():
        p.requires_grad = False

    # Infer output dimensions
    print("Inferring model output dimensions...")
    sample = next(iter(test_loader))
    with torch.no_grad():
        vit_out = vit_model(pixel_values=sample['image'].to(device), output_hidden_states=True)
        vit_logits = vit_out.logits
        vit_features = vit_out.hidden_states[-1][:, 0, :]
        elm_logits = elm_model(vit_features, sample['soil'].to(device))
        lstm_out = lstm_model(sample['weather'].to(device), sample['soil'].to(device))
        if lstm_out.dim() == 1:
            lstm_out = lstm_out.unsqueeze(1)
    
    C = vit_logits.shape[1]
    E = elm_logits.shape[1]
    L = lstm_out.shape[1]
    fusion_input_dim = C + E + L

    # Load fusion head
    fusion = FusionHead(input_dim=fusion_input_dim, hidden_dim=256, num_classes=num_classes, dropout=0.3)
    fusion.to(config['device'])
    fusion.eval()
    
    if Path(config['fusion_checkpoint']).exists():
        ck = torch.load(config['fusion_checkpoint'], map_location=device)
        if isinstance(ck, dict) and 'fusion_state_dict' in ck:
            fusion.load_state_dict(ck['fusion_state_dict'])
        else:
            fusion.load_state_dict(ck)
        print("[+] Fusion checkpoint loaded")
    else:
        print("[-] Fusion checkpoint not found!")

    for p in fusion.parameters():
        p.requires_grad = False

    # Build SoilVisioNet
    model = SoilVisioNet(vit_model=vit_model, lstm_model=lstm_model, elm_model=elm_model, fusion_head=fusion)
    model.to(device)
    model.eval()

    # Inference on test set
    print("\n" + "=" * 70)
    print("Running inference on test set...")
    print("=" * 70 + "\n")

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            images = batch['image'].to(device)
            weather = batch['weather'].to(device)
            soil = batch['soil'].to(device)
            labels = batch['label'].to(device)

            # Measure inference time
            start_time = time.time()
            logits = model(images, weather, soil)
            elapsed = time.time() - start_time
            inference_times.append(elapsed)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    total_inference_time = sum(inference_times)
    avg_inference_time_per_batch = total_inference_time / len(inference_times)
    avg_inference_time_per_sample = avg_inference_time_per_batch / config['batch_size']

    # Print results
    print("\n" + "=" * 70)
    print("SoilVisioNet - TEST SET PERFORMANCE METRICS")
    print("=" * 70)
    print(f"\nAccuracy:                    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score (weighted):         {f1:.4f}")
    print(f"Precision (weighted):        {precision:.4f}")
    print(f"Recall (weighted):           {recall:.4f}")
    print(f"\nInference Time:")
    print(f"  Total inference time:      {total_inference_time:.2f} seconds")
    print(f"  Avg time per batch (n=16): {avg_inference_time_per_batch:.4f} seconds")
    print(f"  Avg time per sample:       {avg_inference_time_per_sample*1000:.2f} milliseconds")
    print(f"\nDataset Size:")
    print(f"  Test samples:              {len(all_labels)}")
    print(f"  Number of disease classes: {num_classes}")
    print("=" * 70)

    # Save results
    results = {
        'model_name': 'SoilVisioNet',
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_set_size': int(len(all_labels)),
        'num_classes': int(num_classes),
        'metrics': {
            'accuracy': float(accuracy),
            'f1_score_weighted': float(f1),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall)
        },
        'inference_time': {
            'total_seconds': float(total_inference_time),
            'per_batch_seconds': float(avg_inference_time_per_batch),
            'per_sample_milliseconds': float(avg_inference_time_per_sample * 1000)
        },
        'batch_size': config['batch_size'],
        'device': config['device']
    }

    results_path = Path('results/hybrid') / 'soilvisionet_evaluation.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Results saved to: {results_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
