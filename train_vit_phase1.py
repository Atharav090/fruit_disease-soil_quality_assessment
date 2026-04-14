"""
Vision Transformer (ViT) Training Pipeline for Disease Detection
Phase 1: Pre-train on PlantVillage + Fine-tune on Fruit Dataset
OPTIMIZED FOR LOW-END SYSTEMS: 16GB RAM, i7 4th Gen CPU, No GPU
Target: Stable training with memory-efficient approach
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our dataset loader
from data_loader import UnifiedDiseaseDataset, get_dataloaders

class ViTTrainer:
    """Train Vision Transformer for disease classification - Memory Optimized"""
    
    model: ViTForImageClassification
    
    def __init__(self, num_classes, output_dir='results/vit_phase1', device='cpu'):
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Load pre-trained ViT
        print("Loading ViT model from HuggingFace...")
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()  # Set to eval mode initially
        
        # use ViTImageProcessor (replaces the deprecated ViTFeatureExtractor)
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224'
        )
        
        print(f"ViT Model loaded (Base, 768-dim embeddings)")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0
        self.best_model_path = self.output_dir / 'best_model.pt'
    
    def train_epoch(self, train_loader, optimizer, criterion, accumulation_steps=2):
        """Train for one epoch with gradient accumulation for memory efficiency"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        accumulation_counter = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=images)
            logits = outputs.logits
            
            # Calculate loss (normalize by accumulation steps)
            loss = criterion(logits, labels) / accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulation_counter += 1
            
            # Optimizer step every accumulation_steps batches
            if accumulation_counter == accumulation_steps or batch_idx == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0
            
            # Track metrics
            total_loss += loss.item() * accumulation_steps
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
            
            # Free memory explicitly
            del images, labels, outputs, logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate model with memory optimization"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Free memory explicitly
                del images, labels, outputs, logits
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=3, learning_rate=5e-5, accumulation_steps=2):
        """Full training loop with memory optimization for low-end systems"""
        
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Learning rate: {learning_rate}")
        print(f"Gradient accumulation steps: {accumulation_steps}")
        print(f"{'='*70}\n")
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 70)
            
            # Train with gradient accumulation
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, accumulation_steps)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, train_acc, val_acc)
                print(f"  [+] New best model saved (Val Acc: {val_acc:.4f})")
        
        print("\n" + "=" * 70)
        print(f"Training complete! Best validation accuracy: {self.best_val_acc:.4f}")
        print("=" * 70)
        
        return self.history
    
    def save_checkpoint(self, epoch, train_acc, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'num_classes': self.num_classes,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.best_model_path)
    
    def load_checkpoint(self):
        """Load best model checkpoint"""
        if self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)  # type: ignore
            print(f"[+] Loaded best model (Val Acc: {checkpoint['val_acc']:.4f})")
            return True
        return False
    
    def evaluate(self, test_loader, disease_labels):
        """Evaluate on test set with detailed metrics and memory optimization"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        print("\nEvaluating on test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Free memory explicitly
                del images, labels, outputs, logits
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Print results
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION RESULTS")
        print("=" * 70)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {self.output_dir / 'test_metrics.json'}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        print(f"[+] Training history saved to {self.output_dir / 'training_history.png'}")
        plt.close()

def main():
    """Main training script optimized for low-end systems (16GB RAM, i7 4th Gen CPU)"""
    
    print("=" * 70)
    print("SOILVISIONET - PHASE 1: VISION TRANSFORMER TRAINING")
    print("LOW-END SYSTEM OPTIMIZATION (16GB RAM, i7 4th Gen, CPU-Only)")
    print("=" * 70)
    
    # Configuration optimized for low-end systems (auto-upgrades with GPU)
    # Device selection: prefer first CUDA device when available
    device_str = 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'

    config = {
        'data_root': 'data/unified_dataset',
        'batch_size': 4,  # Will increase to 16 if GPU available
        'num_workers': 0,  # Safe default; increased when GPU detected
        'num_epochs': 3,
        'learning_rate': 5e-5,
        'image_size': 224,
        'device': device_str,
        'accumulation_steps': 2,
    }

    # Auto-detect GPU and optimize batch size / dataloader settings
    if device_str.startswith('cuda'):
        config['batch_size'] = 16
        config['num_workers'] = 2
        print(f"\n[GPU AUTO-DETECTED] Using {torch.cuda.get_device_name(0)} (device {device_str})")
    else:
        print(f"\n[CPU MODE] Using optimized settings for low-end systems")
    
    print(f"\n{'='*70}")
    print("CONFIGURATION (Optimized for Low-End Systems)")
    print('='*70)
    print(f"\nHardware Configuration:")
    print(f"  RAM: 16GB")
    print(f"  Processor: i7 4th Generation")
    print(f"  GPU: None (CPU-only)")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config['batch_size']} (reduced from 16)")
    print(f"  Epochs: {config['num_epochs']} (reduced from 5)")
    print(f"  Learning rate: {config['learning_rate']} (reduced from 1e-4)")
    print(f"  Device: {config['device']}")
    print(f"  Gradient accumulation: {config['accumulation_steps']} steps")
    print(f"  Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
    print(f"  Image size: {config['image_size']}x{config['image_size']}")
    
    print(f"\nMemory Optimization Features:")
    print(f"  [+] Gradient accumulation (effective batch size without memory spike)")
    print(f"  [+] Explicit memory cleanup between batches")
    print(f"  [+] CPU-only processing (no GPU memory overhead)")
    print(f"  [+] Reduced batch size and epochs")
    print(f"  [+] Lower learning rate for stability")
    
    # Create dataloaders
    print(f"\n{'='*70}")
    print("Loading dataset...")
    print('='*70)
    
    train_loader, val_loader, test_loader, datasets = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=True,
        image_size=config['image_size']
    )
    
    num_classes = datasets['train'].num_classes
    disease_labels = datasets['train'].diseases
    
    # Report final device info
    device = torch.device(config['device'])
    if device.type == 'cuda':
        print(f"[GPU DETECTED] Using {torch.cuda.get_device_name(0)} (device {config['device']})")
        print(f"[GPU] CUDA Version: {torch.version.cuda}")
        print(f"[GPU] Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"[GPU NOT FOUND] Using CPU for training")
    
    # Initialize trainer
    trainer = ViTTrainer(
        num_classes=num_classes,
        output_dir='results/vit_phase1',
        device=config['device']
    )
    
    # Train with gradient accumulation
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        accumulation_steps=config['accumulation_steps']
    )
    
    # Plot training history
    trainer.plot_history()
    
    # Evaluate on test set
    trainer.load_checkpoint()
    results = trainer.evaluate(test_loader, disease_labels)
    
    print("\n" + "=" * 70)
    print("PHASE 1 TRAINING COMPLETE [SUCCESS]")
    print(f"Results saved to: {trainer.output_dir}")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    results = main()
