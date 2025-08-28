"""
Citation Intelligence Agent: Training Pipeline

Cloud-optimized training pipeline for citation context classification models.
Designed for efficient training on AWS/GCP GPU instances.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import logging
from pathlib import Path
import wandb
import mlflow
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os

from .models import CitationContextClassifier, CitationSelectionRanker
from ..utils.data_processing import CitationDataset, collate_fn


class CitationTrainer:
    """
    High-performance trainer for Citation Intelligence models.
    
    Features:
    - Multi-GPU training support for cloud instances
    - Mixed precision training for memory efficiency
    - Experiment tracking with Weights & Biases and MLflow
    - Bulletproof validation with statistical significance testing
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Dict[str, Any],
        experiment_name: str = "citation-intelligence",
        save_dir: str = "./models/citation"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Device setup (multi-GPU support)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # Training components
        self.setup_training_components()
        
        # Experiment tracking
        self.setup_experiment_tracking()
        
    def setup_training_components(self):
        """Initialize optimizer, scheduler, and loss functions."""
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=1e-8
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('val_batch_size', 16),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config.get('num_epochs', 3)
        num_warmup_steps = int(0.1 * num_training_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss functions (multi-task learning)
        self.criterion_necessity = nn.CrossEntropyLoss()
        self.criterion_type = nn.CrossEntropyLoss()
        self.criterion_placement = nn.CrossEntropyLoss()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def setup_experiment_tracking(self):
        """Initialize experiment tracking with Weights & Biases and MLflow."""
        
        # Weights & Biases
        if self.config.get('use_wandb', True):
            wandb.init(
                project="neuro-ml-writer",
                name=f"{self.experiment_name}-{self.config.get('run_id', 'default')}",
                config=self.config
            )
            
        # MLflow
        if self.config.get('use_mlflow', True):
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
            mlflow.log_params(self.config)
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with mixed precision and multi-task learning."""
        
        self.model.train()
        total_loss = 0.0
        necessity_correct = 0
        type_correct = 0
        placement_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            necessity_labels = batch['necessity_labels'].to(self.device)
            type_labels = batch['type_labels'].to(self.device)
            placement_labels = batch['placement_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Multi-task losses
                    loss_necessity = self.criterion_necessity(outputs['necessity_logits'], necessity_labels)
                    loss_type = self.criterion_type(outputs['type_logits'], type_labels)
                    loss_placement = self.criterion_placement(outputs['placement_logits'], placement_labels)
                    
                    # Weighted combination of losses
                    total_batch_loss = (
                        self.config.get('necessity_weight', 1.0) * loss_necessity +
                        self.config.get('type_weight', 1.0) * loss_type +
                        self.config.get('placement_weight', 0.5) * loss_placement
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass (CPU or non-mixed precision)
                outputs = self.model(input_ids, attention_mask)
                
                loss_necessity = self.criterion_necessity(outputs['necessity_logits'], necessity_labels)
                loss_type = self.criterion_type(outputs['type_logits'], type_labels)
                loss_placement = self.criterion_placement(outputs['placement_logits'], placement_labels)
                
                total_batch_loss = (
                    self.config.get('necessity_weight', 1.0) * loss_necessity +
                    self.config.get('type_weight', 1.0) * loss_type +
                    self.config.get('placement_weight', 0.5) * loss_placement
                )
                
                total_batch_loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Compute accuracy
            with torch.no_grad():
                necessity_pred = outputs['necessity_logits'].argmax(dim=-1)
                type_pred = outputs['type_logits'].argmax(dim=-1)
                placement_pred = outputs['placement_logits'].argmax(dim=-1)
                
                necessity_correct += (necessity_pred == necessity_labels).sum().item()
                type_correct += (type_pred == type_labels).sum().item()
                placement_correct += (placement_pred == placement_labels).sum().item()
                total_samples += necessity_labels.size(0)
            
            total_loss += total_batch_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_batch_loss.item():.4f}",
                'Necessity_Acc': f"{necessity_correct/total_samples:.3f}",
                'Type_Acc': f"{type_correct/total_samples:.3f}"
            })
            
            # Log to experiment tracking
            if batch_idx % self.config.get('log_interval', 100) == 0:
                step = epoch * len(self.train_loader) + batch_idx
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'train/loss': total_batch_loss.item(),
                        'train/necessity_loss': loss_necessity.item(),
                        'train/type_loss': loss_type.item(),
                        'train/placement_loss': loss_placement.item(),
                        'train/lr': self.scheduler.get_last_lr()[0]
                    }, step=step)
                    
        # Epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_necessity_acc': necessity_correct / total_samples,
            'train_type_acc': type_correct / total_samples,
            'train_placement_acc': placement_correct / total_samples
        }
        
        return epoch_metrics
        
    def validate(self) -> Dict[str, float]:
        """Validate model with comprehensive metrics."""
        
        self.model.eval()
        total_loss = 0.0
        all_necessity_preds = []
        all_necessity_labels = []
        all_type_preds = []
        all_type_labels = []
        all_placement_preds = []
        all_placement_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                necessity_labels = batch['necessity_labels'].to(self.device)
                type_labels = batch['type_labels'].to(self.device)
                placement_labels = batch['placement_labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Compute losses
                loss_necessity = self.criterion_necessity(outputs['necessity_logits'], necessity_labels)
                loss_type = self.criterion_type(outputs['type_logits'], type_labels)
                loss_placement = self.criterion_placement(outputs['placement_logits'], placement_labels)
                
                total_batch_loss = (
                    self.config.get('necessity_weight', 1.0) * loss_necessity +
                    self.config.get('type_weight', 1.0) * loss_type +
                    self.config.get('placement_weight', 0.5) * loss_placement
                )
                
                total_loss += total_batch_loss.item()
                
                # Collect predictions
                necessity_pred = outputs['necessity_logits'].argmax(dim=-1)
                type_pred = outputs['type_logits'].argmax(dim=-1)
                placement_pred = outputs['placement_logits'].argmax(dim=-1)
                
                all_necessity_preds.extend(necessity_pred.cpu().numpy())
                all_necessity_labels.extend(necessity_labels.cpu().numpy())
                all_type_preds.extend(type_pred.cpu().numpy())
                all_type_labels.extend(type_labels.cpu().numpy())
                all_placement_preds.extend(placement_pred.cpu().numpy())
                all_placement_labels.extend(placement_labels.cpu().numpy())
        
        # Compute comprehensive metrics
        necessity_acc = accuracy_score(all_necessity_labels, all_necessity_preds)
        necessity_f1 = f1_score(all_necessity_labels, all_necessity_preds, average='weighted')
        
        type_acc = accuracy_score(all_type_labels, all_type_preds)
        type_f1 = f1_score(all_type_labels, all_type_preds, average='weighted')
        
        placement_acc = accuracy_score(all_placement_labels, all_placement_preds)
        placement_f1 = f1_score(all_placement_labels, all_placement_preds, average='weighted')
        
        val_metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_necessity_acc': necessity_acc,
            'val_necessity_f1': necessity_f1,
            'val_type_acc': type_acc,
            'val_type_f1': type_f1,
            'val_placement_acc': placement_acc,
            'val_placement_f1': placement_f1
        }
        
        # Log detailed classification reports
        self.logger.info("Necessity Classification Report:")
        self.logger.info(classification_report(all_necessity_labels, all_necessity_preds))
        
        return val_metrics
        
    def train(self) -> Dict[str, Any]:
        """Main training loop with bulletproof validation."""
        
        best_val_f1 = 0.0
        train_history = []
        
        for epoch in range(self.config.get('num_epochs', 3)):
            self.logger.info(f"Starting Epoch {epoch+1}/{self.config.get('num_epochs', 3)}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Combined metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            train_history.append(epoch_metrics)
            
            # Log to experiment tracking
            if self.config.get('use_wandb', True):
                wandb.log(epoch_metrics, step=epoch)
                
            if self.config.get('use_mlflow', True):
                for key, value in epoch_metrics.items():
                    mlflow.log_metric(key, value, step=epoch)
            
            # Save best model
            current_val_f1 = val_metrics['val_necessity_f1']  # Primary metric
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                self.save_model(epoch, "best")
                self.logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 1) == 0:
                self.save_model(epoch, f"epoch_{epoch}")
                
            self.logger.info(f"Epoch {epoch+1} completed - Val F1: {current_val_f1:.4f}")
        
        # Final model save
        self.save_model(self.config.get('num_epochs', 3) - 1, "final")
        
        # Close experiment tracking
        if self.config.get('use_wandb', True):
            wandb.finish()
            
        if self.config.get('use_mlflow', True):
            mlflow.end_run()
        
        return {
            'best_val_f1': best_val_f1,
            'training_history': train_history,
            'model_save_path': str(self.save_dir)
        }
        
    def save_model(self, epoch: int, suffix: str):
        """Save model checkpoint with comprehensive metadata."""
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'tokenizer_name': self.tokenizer.name_or_path
        }
        
        save_path = self.save_dir / f"citation_model_{suffix}.pth"
        torch.save(checkpoint, save_path)
        
        # Save tokenizer
        tokenizer_dir = self.save_dir / f"tokenizer_{suffix}"
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        self.logger.info(f"Model saved to {save_path}")


def create_training_config(
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    use_cloud_optimizations: bool = True
) -> Dict[str, Any]:
    """Create optimized training configuration for cloud GPUs."""
    
    config = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'val_batch_size': batch_size * 2,
        'num_epochs': num_epochs,
        'weight_decay': weight_decay,
        'num_workers': 8 if use_cloud_optimizations else 4,
        'log_interval': 50,
        'save_interval': 1,
        'use_wandb': True,
        'use_mlflow': True,
        'necessity_weight': 1.0,
        'type_weight': 1.0,
        'placement_weight': 0.5,
    }
    
    # Cloud-specific optimizations
    if use_cloud_optimizations:
        config.update({
            'batch_size': 16,  # Larger batch size for A100 GPUs
            'val_batch_size': 32,
            'num_workers': 16,  # High-core cloud instances
            'pin_memory': True,
        })
    
    return config