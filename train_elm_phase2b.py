"""
ELM (Extreme Learning Machine) Fusion for SoilVisioNet
Phase 2B: Fuse ViT features + Soil parameters
Expected improvement: +3-6% accuracy
Fast training: <1 hour
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from tqdm import tqdm
import json
from datetime import datetime

from data_loader import UnifiedDiseaseDataset, get_dataloaders

class ELMHiddenLayer(nn.Module):
    """ELM Hidden Layer with random weights"""
    
    def __init__(self, input_size, hidden_size, activation='relu'):
        super(ELMHiddenLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Random weights (NOT trained)
        self.weight = nn.Parameter(
            torch.randn(input_size, hidden_size) * 0.1,
            requires_grad=False
        )
        
        self.bias = nn.Parameter(
            torch.randn(hidden_size) * 0.1,
            requires_grad=False
        )
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_size)
        
        Returns:
            h: (batch_size, hidden_size)
        """
        # Ensure input is float32 and move parameters to input device
        x = x.float()
        device = x.device
        weight = self.weight.to(device)
        bias = self.bias.to(device)

        # Linear transformation with fixed random weights
        z = torch.matmul(x, weight) + bias
        # Apply activation
        h = self.activation(z)
        return h

class SoilVisioNetELM(nn.Module):
    """
    ELM Fusion of ViT features + Soil parameters
    """
    
    def __init__(self, vit_feature_dim=768, soil_dim=4, hidden_size=512, 
                 num_classes=50):
        super(SoilVisioNetELM, self).__init__()
        
        self.vit_feature_dim = vit_feature_dim
        self.soil_dim = soil_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Total input size: ViT features + Soil parameters
        input_size = vit_feature_dim + soil_dim
        
        # ELM Hidden Layer (random, not trained)
        self.elm_hidden = ELMHiddenLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            activation='relu'
        )
        
        # Output layer (to be trained with least-squares)
        # Will be initialized later as a torch tensor
        self.output_weight = torch.nn.Parameter(torch.zeros(hidden_size, num_classes))
        self.output_bias = torch.nn.Parameter(torch.zeros(num_classes))
        
    def forward_hidden(self, vit_features, soil_features):
        """
        Get hidden layer output (this is used for training output layer)
        
        Args:
            vit_features: (batch_size, 768) from ViT
            soil_features: (batch_size, 4)
        
        Returns:
            h: (batch_size, hidden_size) - Hidden layer output
        """
        # Concatenate features and ensure float32
        # Move both inputs to the same device (prefer vit_features device)
        device = vit_features.device if hasattr(vit_features, 'device') else None
        vit = vit_features.float().to(device)
        soil = soil_features.float().to(device)
        x = torch.cat([vit, soil], dim=1)  # (batch, 772)

        # ELM hidden layer (ELMHiddenLayer will handle its params device-safely)
        h = self.elm_hidden(x)
        
        return h
    
    def forward(self, vit_features, soil_features):
        """
        Forward pass with output layer
        
        Args:
            vit_features: (batch_size, 768)
            soil_features: (batch_size, 4)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        h = self.forward_hidden(vit_features, soil_features)

        # Ensure output weights and bias are on the same device as h
        device = h.device
        out_w = self.output_weight.to(device)
        out_b = self.output_bias.to(device)

        # Apply trained output layer
        logits = torch.matmul(h, out_w) + out_b

        return logits

class ELMFusionTrainer:
    """Train ELM fusion layer"""
    
    def __init__(self, vit_model, num_classes, output_dir='results/elm_phase2b',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.vit_model = vit_model
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Create ELM model
        self.elm_model = SoilVisioNetELM(
            vit_feature_dim=768,
            soil_dim=4,
            hidden_size=512,
            num_classes=num_classes
        )
        self.elm_model = self.elm_model.to(device)
        
        print(f"ELM Model created")
        print(f"  Input dim: 768 (ViT) + 4 (Soil) = 772")
        print(f"  Hidden dim: 512")
        print(f"  Output dim: {num_classes}")
        print(f"  Training method: Least-squares (random features fixed)")
    
    def extract_vit_features(self, dataloader):
        """
        Extract ViT features for entire dataset
        These are fixed (not trained further)
        """
        self.vit_model.eval()
        
        all_features = []
        all_soil = []
        all_labels = []
        
        print("\nExtracting ViT features...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['image'].to(self.device)
                soil = batch['soil'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get ViT features (use last hidden state before classification head)
                outputs = self.vit_model(pixel_values=images, output_hidden_states=True)
                # Shape: (batch_size, 768)
                features = outputs.hidden_states[-1][:, 0, :]  # CLS token
                
                all_features.append(features.cpu())
                all_soil.append(soil.cpu())
                all_labels.append(labels.cpu())
        
        features = torch.cat(all_features, dim=0)  # (N, 768)
        soil = torch.cat(all_soil, dim=0)          # (N, 4)
        labels = torch.cat(all_labels, dim=0)       # (N,)
        
        print(f"  Features shape: {features.shape}")
        print(f"  Soil shape: {soil.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        return features, soil, labels
    
    def train_output_layer(self, train_features, train_soil, train_labels):
        """
        Train output layer using least-squares regression
        This is fast: <1 minute
        """
        print("\nTraining output layer (Least-Squares)...")
        
        # Get hidden layer output
        with torch.no_grad():
            h = self.elm_model.forward_hidden(train_features, train_soil)
            h = h.cpu().numpy()
        
        # Target
        y = train_labels.numpy()
        
        # Convert to one-hot
        y_onehot = np.eye(self.num_classes)[y]
        
        # Least-squares solve: min ||Hw - y||^2
        # Using Ridge regression for regularization
        ridge = Ridge(alpha=0.01, solver='auto')
        ridge.fit(h, y_onehot)
        
        # Get output weights
        w = ridge.coef_.T  # (hidden_size, num_classes)
        b = np.atleast_1d(ridge.intercept_)  # (num_classes,)
        
        print(f"  Output weight shape: {w.shape}")
        print(f"  Output bias shape: {b.shape}")
        
        # Store in model
        output_weight = torch.from_numpy(w).float().to(self.device)
        output_bias = torch.from_numpy(b).float().to(self.device)
        
        self.elm_model.output_weight.data = output_weight
        self.elm_model.output_bias.data = output_bias
        
        print("[+] Output layer trained")
        
        return ridge
    
    def evaluate(self, features, soil, labels, split_name='val'):
        """Evaluate model"""
        self.elm_model.eval()
        
        with torch.no_grad():
            logits = self.elm_model(
                features.to(self.device),
                soil.to(self.device)
            )
            logits = logits.cpu()
        
        preds = torch.argmax(logits, dim=1).numpy()
        labels_np = labels.numpy()
        
        accuracy = np.mean(preds == labels_np)
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def train(self, train_loader, val_loader, test_loader, datasets):
        """Full training pipeline"""
        
        print("\n" + "=" * 70)
        print("PHASE 2B: ELM FUSION TRAINING")
        print("=" * 70)
        
        # Step 1: Extract features from all data
        print("\nStep 1: Extract ViT features")
        train_features, train_soil, train_labels = self.extract_vit_features(train_loader)
        val_features, val_soil, val_labels = self.extract_vit_features(val_loader)
        test_features, test_soil, test_labels = self.extract_vit_features(test_loader)
        
        # Step 2: Train output layer on training data
        print("\nStep 2: Train output layer")
        self.train_output_layer(train_features, train_soil, train_labels)
        
        # Step 3: Evaluate
        print("\nStep 3: Evaluate on all splits")
        train_acc = self.evaluate(train_features, train_soil, train_labels, 'train')
        val_acc = self.evaluate(val_features, val_soil, val_labels, 'val')
        test_acc = self.evaluate(test_features, test_soil, test_labels, 'test')
        
        # Save results
        metrics = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'improvement_over_vit': float(test_acc - 0.92),  # Assuming ViT gets ~92%
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'elm_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model
        torch.save(self.elm_model.state_dict(), self.output_dir / 'elm_model.pt')
        
        print("\n" + "=" * 70)
        print("PHASE 2B COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
        
        return metrics

def main():
    """Main script"""
    
    print("=" * 70)
    print("PHASE 2B: ELM FUSION FOR SOILVISIONET")
    print("=" * 70)
    
    # Configuration (auto-optimizes with GPU)
    device_str = 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'

    config = {
        'data_root': 'data/unified_dataset',
        'batch_size': 4,
        'num_workers': 0,
        'device': device_str,
        'vit_checkpoint': 'results/vit_phase1/best_model.pt'
    }

    # Auto-detect GPU
    if device_str.startswith('cuda'):
        config['batch_size'] = 16
        config['num_workers'] = 2
        print(f"\n[GPU DETECTED] {torch.cuda.get_device_name(0)} (device {device_str})")
    else:
        print(f"\n[CPU MODE] ELM training on CPU")
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    num_classes = datasets['train'].num_classes
    
    # Load pre-trained ViT
    print("\nLoading ViT model...")
    from transformers import ViTForImageClassification, ViTImageProcessor
    
    # Create model with correct number of classes (skip pretrained download)
    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Allow loading 55-class checkpoint into 1000-class model
    )
    
    # Load checkpoint if exists
    if Path(config['vit_checkpoint']).exists():
        checkpoint = torch.load(config['vit_checkpoint'], map_location=config['device'])
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[+] Loaded ViT checkpoint")
    else:
        print("[-] No ViT checkpoint found. Using pre-trained ViT.")
        print("    Run train_vit_phase1.py first for best results.")

    # Move model to selected device
    vit_model = vit_model.to(torch.device(config['device']))  # type: ignore
    vit_model.eval()
    
    # Train ELM
    trainer = ELMFusionTrainer(
        vit_model=vit_model,
        num_classes=num_classes,
        output_dir='results/elm_phase2b',
        device=config['device']
    )
    
    metrics = trainer.train(train_loader, val_loader, test_loader, datasets)
    
    print("\nFinal Metrics:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    
    return metrics

if __name__ == "__main__":
    metrics = main()
