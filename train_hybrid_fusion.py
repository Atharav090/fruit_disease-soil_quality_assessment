"""
Hybrid Fusion Training: SoilVisioNet
Combines ViT + LSTM + ELM by freezing pretrained models
and training a lightweight fusion head (late fusion).

Usage: python train_hybrid_fusion.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Import dataset and existing model classes
from data_loader import get_dataloaders
from train_elm_phase2b import SoilVisioNetELM
from train_vit_phase1 import ViTTrainer
from train_lstm_phase2a import WeatherLSTM
from transformers import ViTForImageClassification


class FusionHead(nn.Module):
    """Small trainable fusion head. Accepts concatenated logits/features."""
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
    """Wrapper that runs ViT, LSTM, and ELM and forwards their outputs to fusion head."""
    def __init__(self, vit_model, lstm_model, elm_model, fusion_head):
        super(SoilVisioNet, self).__init__()
        self.vit = vit_model
        self.lstm = lstm_model
        self.elm = elm_model
        self.fusion = fusion_head

    def forward(self, images, weather, soil):
        # ViT: get logits and features
        # Ensure output_hidden_states=True to extract CLS features
        vit_outputs = self.vit(pixel_values=images, output_hidden_states=True)
        try:
            vit_logits = vit_outputs.logits
        except AttributeError:
            vit_logits = vit_outputs[0]
        # CLS features (last hidden state, token 0)
        vit_features = vit_outputs.hidden_states[-1][:, 0, :]

        # ELM expects vit_features + soil -> logits
        elm_logits = self.elm(vit_features, soil)

        # LSTM: may accept (weather, soil) and output logits or single value
        lstm_out = self.lstm(weather, soil)
        # If lstm_out is (batch, 1) or (batch,), keep as (batch, L)
        if lstm_out.dim() == 1:
            lstm_logits = lstm_out.unsqueeze(1).float()
        else:
            lstm_logits = lstm_out.float()

        # Prepare tensors to concatenate: vit_logits (B,C), elm_logits (B,C), lstm_logits (B,L)
        x = torch.cat([vit_logits, elm_logits, lstm_logits], dim=1)
        return self.fusion(x)


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def main():
    device_str = 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'

    config = {
        'data_root': 'data/unified_dataset',
        'batch_size': 16,
        'num_workers': 0,
        'num_epochs': 5,
        'learning_rate': 1e-3,
        'device': device_str,
        'vit_checkpoint': 'results/vit_phase1/best_model.pt',
        'lstm_checkpoint': 'results/lstm_phase2a/best_lstm_model.pt',
        'elm_checkpoint': 'results/elm_phase2b/elm_model.pt'
    }

    # Auto-detect GPU
    if device_str.startswith('cuda'):
        config['num_workers'] = 2
        print(f"\n[GPU DETECTED] {torch.cuda.get_device_name(0)} (device {device_str})")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        config['batch_size'] = 8  # Reduce for CPU
        print(f"\n[CPU MODE] Hybrid fusion training on CPU")

    device = torch.device(config['device'])

    print("Loading dataset...")
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    num_classes = datasets['train'].num_classes

    # Load ViT model
    print("Loading ViT model...")
    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    if Path(config['vit_checkpoint']).exists():
        ck = torch.load(config['vit_checkpoint'], map_location=device)
        # expected saved checkpoint format may include 'model_state_dict'
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            vit_model.load_state_dict(ck['model_state_dict'])
        else:
            try:
                vit_model.load_state_dict(ck)
            except Exception:
                print('Warning: could not load ViT checkpoint exactly; continuing with pretrained ViT')
    vit_model.to(config['device'])
    vit_model.eval()
    freeze_model(vit_model)

    # Load LSTM model
    print("Loading LSTM model...")
    # Create WeatherLSTM with num_classes for compatibility; it will load whatever shape in checkpoint
    lstm_model = WeatherLSTM(num_classes=num_classes, hidden_size=256, num_layers=2, dropout=0.3)
    if Path(config['lstm_checkpoint']).exists():
        ck = torch.load(config['lstm_checkpoint'], map_location=device)
        try:
            lstm_model.load_state_dict(ck)
        except Exception:
            # try loading nested dict
            if isinstance(ck, dict) and 'model_state_dict' in ck:
                lstm_model.load_state_dict(ck['model_state_dict'])
            else:
                print('Warning: could not load LSTM checkpoint exactly; continuing with initialized LSTM')
    lstm_model.to(config['device'])
    freeze_model(lstm_model)

    # Load ELM model
    print("Loading ELM model...")
    elm_model = SoilVisioNetELM(vit_feature_dim=768, soil_dim=4, hidden_size=512, num_classes=num_classes)
    if Path(config['elm_checkpoint']).exists():
        try:
            elm_state = torch.load(config['elm_checkpoint'], map_location=device)
            elm_model.load_state_dict(elm_state)
        except Exception:
            print('Warning: could not load ELM checkpoint exactly; continuing with initialized ELM')
    elm_model.to(config['device'])
    freeze_model(elm_model)

    # Inspect output dims by running one batch through the frozen models
    print('Inferring output dims from one batch...')
    sample = next(iter(train_loader))
    images = sample['image'].to(device)
    weather = sample['weather'].to(device)
    soil = sample['soil'].to(device)

    with torch.no_grad():
        vit_out = vit_model(pixel_values=images, output_hidden_states=True)
        vit_logits = vit_out.logits
        vit_features = vit_out.hidden_states[-1][:, 0, :]
        elm_logits = elm_model(vit_features, soil)
        lstm_out = lstm_model(weather, soil)
        if lstm_out.dim() == 1:
            lstm_out = lstm_out.unsqueeze(1)

    C = vit_logits.shape[1]
    E = elm_logits.shape[1]
    L = lstm_out.shape[1]
    print(f'  ViT logits dim: {C}, ELM logits dim: {E}, LSTM out dim: {L}')

    # Build fusion head
    input_dim = C + E + L
    fusion = FusionHead(input_dim=input_dim, hidden_dim=256, num_classes=num_classes, dropout=0.3).to(device)

    # Create SoilVisioNet wrapper
    model = SoilVisioNet(vit_model=vit_model, lstm_model=lstm_model, elm_model=elm_model, fusion_head=fusion)
    model = model.to(device)

    # Only train fusion parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    best_path = Path('results/hybrid')
    best_path.mkdir(parents=True, exist_ok=True)

    # Training loop (fusion head only)
    print('\nStarting fusion training...')
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}')
        for batch in pbar:
            images = batch['image'].to(device)
            weather = batch['weather'].to(device)
            soil = batch['soil'].to(device)
            labels = batch['label'].to(device)

            logits = model(images, weather, soil)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        val_acc = evaluate(model, val_loader, device)

        print(f'Epoch {epoch+1}: Train Loss {total_loss/len(train_loader):.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}')

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                'fusion_state_dict': fusion.state_dict(),
                'timestamp': datetime.now().isoformat()
            }, best_path / 'best_hybrid_fusion.pt')
            print('[+] Best fusion saved')

    # Final test
    print('\nEvaluating best fusion on test set...')
    fusion_ck = best_path / 'best_hybrid_fusion.pt'
    if fusion_ck.exists():
        data = torch.load(fusion_ck, map_location=device)
        fusion.load_state_dict(data['fusion_state_dict'])
    test_acc = evaluate(model, test_loader, device, split_name='test')
    print(f'Final Test Accuracy: {test_acc:.4f}')

    # Save metrics
    metrics = {'best_val_acc': float(best_val), 'test_acc': float(test_acc), 'timestamp': datetime.now().isoformat()}
    with open(best_path / 'hybrid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\nDone.')


def evaluate(model, loader, device, split_name='val'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            weather = batch['weather'].to(device)
            soil = batch['soil'].to(device)
            labels = batch['label'].to(device)

            logits = model(images, weather, soil)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f'{split_name.capitalize()} Accuracy: {acc:.4f}')
    return acc


if __name__ == '__main__':
    main()
