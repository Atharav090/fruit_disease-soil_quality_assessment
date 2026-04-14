"""
LSTM Training for Temporal Weather Analysis
Phase 2A: Optional enhancement (+5% accuracy potential)
Processes 30-day weather sequences for disease prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data_loader import UnifiedDiseaseDataset, get_dataloaders

class WeatherLSTM(nn.Module):
    """LSTM for processing 30-day weather sequences with improved architecture"""
    
    def __init__(self, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        """
        Args:
            num_classes: Number of disease classes
            hidden_size: LSTM hidden dimension (256 for better feature extraction)
            num_layers: Number of LSTM layers (2 for better temporal learning)
            dropout: Dropout rate (0.3 for regularization)
        """
        super(WeatherLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with 2 layers for deeper temporal feature extraction
        # Input: (batch, sequence_length=30, features=3) -> (temp, rainfall, humidity)
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Process sequence forwards and backwards
        )
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        
        # Dense layers with better capacity
        # We'll concatenate soil features (4 dims) to LSTM representation
        # LSTM output: (batch, hidden_size * 2) due to bidirectional
        self.fc1 = nn.Linear(hidden_size * 2 + 4, 512)
        self.ln2 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(256, 1)  # Binary output for disease risk prediction
        
        self.relu = nn.ReLU()
        
    def forward(self, weather_seq, soil=None):
        """
        Args:
            weather_seq: (batch_size, 30, 3) - 30 days, 3 features each
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(weather_seq)
        # lstm_out shape: (batch, 30, hidden_size*2)
        
        # Use last time step and apply layer norm
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        last_hidden = self.ln1(last_hidden)
        
        # Concatenate soil features if provided
        if soil is not None:
            # soil expected shape: (batch, 4)
            soil = soil.float()
            x_in = torch.cat([last_hidden, soil], dim=1)
        else:
            # pad zeros for soil when not provided
            zeros = torch.zeros(last_hidden.size(0), 4, device=last_hidden.device, dtype=last_hidden.dtype)
            x_in = torch.cat([last_hidden, zeros], dim=1)

        x = self.relu(self.fc1(x_in))
        x = self.ln2(x)
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.ln3(x)
        x = self.dropout2(x)
        
        logits = self.fc3(x)
        
        return logits

# ... existing WeatherLSTM class stays the same ...

class LSTMTrainer:
    """Train LSTM for weather-based disease prediction"""
    
    def __init__(self, num_classes, output_dir='results/lstm_phase2a',
                 device='cpu', class_weights=None):
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.class_weights = class_weights
        
        # Create model
        self.model = WeatherLSTM(
            num_classes=num_classes,
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        )
        self.model = self.model.to(device)
        
        print(f"Device: {device}")
        print(f"LSTM Model created")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0
        self.best_model_path = self.output_dir / 'best_lstm_model.pt'
        self.accumulation_steps = 2
        self.accumulation_counter = 0
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        self.accumulation_counter = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            weather_seq = batch['weather'].to(self.device)  # (batch, 30, 3)
            soil = batch['soil'].to(self.device)
            labels = batch['risk'].to(self.device).float().unsqueeze(1)  # Convert to (batch, 1) float
            
            # Check if weather is all zeros (debug)
            if torch.all(weather_seq == 0):
                print(f"\nWARNING: All weather values are zero in batch {batch_idx}!")
            
            # Forward pass
            logits = self.model(weather_seq, soil)
            loss = criterion(logits, labels) / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            self.accumulation_counter += 1
            
            if self.accumulation_counter == self.accumulation_steps or batch_idx == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                self.accumulation_counter = 0
            
            # Metrics
            total_loss += loss.item() * self.accumulation_steps
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # Remove output dimension only
            all_preds.extend(preds.cpu().detach().numpy().tolist())
            all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
            
            progress_bar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.4f}'})
            
            del weather_seq, soil, labels, logits, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                weather_seq = batch['weather'].to(self.device)
                soil = batch['soil'].to(self.device)
                labels = batch['risk'].to(self.device).float().unsqueeze(1)  # Convert to (batch, 1) float

                logits = self.model(weather_seq, soil)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # Remove output dimension only
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
                
                del weather_seq, soil, labels, logits, loss
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
        """Full training loop"""
        
        print(f"\nTraining LSTM for {num_epochs} epochs")
        print(f"Learning rate: {learning_rate}")
        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        print("=" * 70)
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3
        )
        
        def lr_lambda(epoch):
            # Simple exponential decay to avoid division by zero
            return 0.95 ** epoch
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # Use BCEWithLogitsLoss for binary disease risk prediction
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            scheduler.step()
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  [+] New best model saved (Val Acc: {val_acc:.4f})")
        
        print("\n" + "=" * 70)
        print(f"Training complete! Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                weather_seq = batch['weather'].to(self.device)
                soil = batch['soil'].to(self.device)
                labels = batch['risk'].to(self.device).float()  # Use 'risk' key

                logits = self.model(weather_seq, soil)
                preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)  # Remove output dimension only
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                del weather_seq, soil, labels, logits
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\nLSTM Test Accuracy: {accuracy:.4f}")
        
        metrics = {
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'lstm_test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return accuracy
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('LSTM Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('LSTM Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lstm_training_history.png', dpi=150)
        print(f"[+] Training history saved")
        plt.close()

def main():
    """Main training script"""
    
    print("=" * 70)
    print("PHASE 2A: LSTM WEATHER ANALYSIS TRAINING")
    print("=" * 70)
    
    device_str = 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'

    config = {
        'data_root': 'data/unified_dataset',
        'batch_size': 4,
        'num_workers': 0,
        'num_epochs': 2,
        'learning_rate': 0.005,
        'device': device_str
    }

    # Auto-detect GPU and optimize
    if device_str.startswith('cuda'):
        config['batch_size'] = 16
        config['num_workers'] = 2
        print(f"\n[GPU DETECTED] {torch.cuda.get_device_name(0)} (device {device_str})")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\n[CPU MODE] LSTM training will run on CPU")
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    num_classes = datasets['train'].num_classes
    
    # Compute class weights from training set to address imbalance
    # Use numpy arrays to avoid pandas ExtensionArray operator issues
    train_labels = datasets['train'].metadata['label'].fillna(-1).to_numpy(dtype=np.int64)
    # ensure non-negative and within range
    valid_mask = (train_labels >= 0)
    counts = np.bincount(train_labels[valid_mask].astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = (1.0 / counts)
    # normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes

    # Train LSTM
    trainer = LSTMTrainer(
        num_classes=num_classes,
        output_dir='results/lstm_phase2a',
        device=config['device'],
        class_weights=class_weights
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate']
    )
    
    trainer.plot_history()
    
    # Load best model and evaluate
    trainer.model.load_state_dict(torch.load(trainer.best_model_path))
    test_acc = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 70)
    print("PHASE 2A COMPLETE")
    print(f"Results saved to: {trainer.output_dir}")
    print("=" * 70)
    
    return test_acc

if __name__ == "__main__":
    test_acc = main()