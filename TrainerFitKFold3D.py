"""
K-Fold Cross Validation Training Script for 3D AMD Grading

This script performs 5-fold cross validation training on 3D OCT data using config_3d.toml
and saves comprehensive validation metrics to logs/5-k-validation_3d/.
"""

import os
import json
import toml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import functional as FM
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from DataPreprocessing import myDataset_mat
from NetModule_3D import NetModule
from Utils.utils import k_fold_split, read_csv_file


class KFold3DDataModule(L.LightningDataModule):
    """DataModule for K-fold training with 3D OCT .mat data."""
    
    def __init__(self, train_indices, val_indices, mat_list, label_list, 
                 img_size, batch_size, shuffle=True):
        super().__init__()
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.mat_list = mat_list
        self.label_list = label_list
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        # Create training lists
        train_mat_list = [self.mat_list[i] for i in self.train_indices]
        train_label_list = [self.label_list[i] for i in self.train_indices]
        
        # Create validation lists
        val_mat_list = [self.mat_list[i] for i in self.val_indices]
        val_label_list = [self.label_list[i] for i in self.val_indices]
        
        self.train_dataset = myDataset_mat(
            train_mat_list, train_label_list, self.img_size, data_type="3d"
        )
        self.val_dataset = myDataset_mat(
            val_mat_list, val_label_list, self.img_size, data_type="3d"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            num_workers=2,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


def compute_validation_metrics(y_true, y_pred, y_probs, num_classes=4):
    """Compute comprehensive validation metrics similar to PredictionVal.py"""
    class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # AUC-ROC scores
    try:
        auc_macro = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
        auc_micro = roc_auc_score(y_true, y_probs, average='micro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_true, y_probs, average='weighted', multi_class='ovr')
        # Per-class AUC using one-vs-rest approach
        auc_per_class = []
        for i in range(num_classes):
            # Create binary labels for class i vs rest
            y_binary = (y_true == i).astype(int)
            if len(np.unique(y_binary)) > 1:  # Check if both classes are present
                auc_per_class.append(roc_auc_score(y_binary, y_probs[:, i]))
            else:
                auc_per_class.append(0.0)
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc_macro = auc_micro = auc_weighted = 0.0
        auc_per_class = [0.0] * num_classes
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro,
        'auc_micro': auc_micro,
        'auc_weighted': auc_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'auc_per_class': auc_per_class,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'class_names': class_names
    }
    
    return results


def save_fold_results(results, fold_idx, output_dir):
    """Save validation results for a single fold"""
    fold_dir = Path(output_dir) / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    json_results = {
        'y_true': results['y_true'].tolist(),
        'y_pred': results['y_pred'].tolist(),
        'y_probs': results['y_probs'].tolist(),
    }
    with open(fold_dir / 'validation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save metrics CSV
    metrics_data = []
    
    # Overall metrics
    metrics_data.extend([
        ['Accuracy', 'Overall', 'All', results['accuracy']],
        ['Precision', 'Macro', 'All', results['precision_macro']],
        ['Precision', 'Micro', 'All', results['precision_micro']],
        ['Precision', 'Weighted', 'All', results['precision_weighted']],
        ['Recall', 'Macro', 'All', results['recall_macro']],
        ['Recall', 'Micro', 'All', results['recall_micro']],
        ['Recall', 'Weighted', 'All', results['recall_weighted']],
        ['F1-Score', 'Macro', 'All', results['f1_macro']],
        ['F1-Score', 'Micro', 'All', results['f1_micro']],
        ['F1-Score', 'Weighted', 'All', results['f1_weighted']],
        ['AUC-ROC', 'Macro', 'All', results['auc_macro']],
        ['AUC-ROC', 'Micro', 'All', results['auc_micro']],
        ['AUC-ROC', 'Weighted', 'All', results['auc_weighted']],
    ])
    
    # Per-class metrics
    for i, class_name in enumerate(results['class_names']):
        metrics_data.extend([
            ['Precision', 'Per-Class', class_name, results['precision_per_class'][i]],
            ['Recall', 'Per-Class', class_name, results['recall_per_class'][i]],
            ['F1-Score', 'Per-Class', class_name, results['f1_per_class'][i]],
            ['AUC-ROC', 'Per-Class', class_name, results['auc_per_class'][i]],
        ])
    
    df_metrics = pd.DataFrame(metrics_data, columns=['Metric', 'Type', 'Class', 'Value'])
    df_metrics.to_csv(fold_dir / 'validation_metrics.csv', index=False)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=results['class_names'], yticklabels=results['class_names'])
    plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(fold_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary text
    with open(fold_dir / 'validation_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"FOLD {fold_idx + 1} VALIDATION EVALUATION REPORT - 3D MODEL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(results['y_true'])}\n")
        f.write(f"Number of Classes: {len(results['class_names'])}\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {results['f1_macro']:.4f}\n")
        f.write(f"Micro F1-Score: {results['f1_micro']:.4f}\n")
        f.write(f"Weighted F1-Score: {results['f1_weighted']:.4f}\n")
        f.write(f"Macro AUC-ROC: {results['auc_macro']:.4f}\n\n")
        
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<16} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}\n")
        f.write("-" * 70 + "\n")
        for i, class_name in enumerate(results['class_names']):
            f.write(f"{class_name:<16} {results['precision_per_class'][i]:<10.4f} "
                   f"{results['recall_per_class'][i]:<10.4f} {results['f1_per_class'][i]:<10.4f} "
                   f"{results['auc_per_class'][i]:<10.4f}\n")


def evaluate_fold(model, dataloader, device, num_classes=4):
    """Evaluate 3D model on validation data and return predictions and probabilities"""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            mat_data, targets, _ = batch
            mat_data, targets = mat_data.to(device), targets.to(device)
            
            logits = model(mat_data)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def run_kfold_training(config_path=None):
    """Main function to run k-fold cross validation training with comprehensive validation.
    
    Args:
        config_path (str, optional): Path to config file. If None, defaults to config_3d.toml
    """
    # Set random seed
    L.seed_everything(1234)
    
    # Determine config path
    if config_path is None:
        config_path = 'configs/config_3d.toml'
    
    print(f"Loading 3D configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Get configuration parameters
    data_config = config['DataModule']
    net_config = config['NetModule']
    
    # Load 3D data from CSV
    data_csv = read_csv_file(data_config['label_path'])
    mat_list = [os.path.join(data_config['data_path'], row['caseID'] + '.mat') for row in data_csv]
    label_list = [int(row['label']) for row in data_csv]
    
    if len(mat_list) != len(label_list):
        raise ValueError(f"Mismatch: {len(mat_list)} mat files vs {len(label_list)} labels")
    
    # K-fold parameters
    k_folds = 5  # Fixed to 5-fold as requested
    img_size = tuple(data_config['image_shape'])  # (D, H, W) for 3D
    batch_size = data_config['batch_size']
    num_classes = data_config['n_class']
    
    # Generate k-fold splits
    _, kfold_indices = k_fold_split(list(range(len(label_list))), fold=k_folds)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"logs/5-k-validation_3d/{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting {k_folds}-fold cross validation for 3D AMD Grading")
    print(f"Total samples: {len(label_list)}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Results will be saved to: {output_dir}")
    print("-" * 70)
    
    all_fold_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train each fold
    for fold_idx, (train_indices, val_indices) in enumerate(kfold_indices):
        print(f"\nTraining Fold {fold_idx + 1}/{k_folds}")
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        # Create data module for this fold
        datamodule = KFold3DDataModule(
            train_indices=train_indices,
            val_indices=val_indices,
            mat_list=mat_list,
            label_list=label_list,
            img_size=img_size,
            batch_size=batch_size,
            shuffle=data_config.get('shuffle', True)
        )
        
        # Update config with current fold info
        fold_config = config.copy()
        fold_config['NetModule']['checkpoint_dir'] = f"./logs/5-k-validation_3d/{timestamp}/fold_{fold_idx + 1}/"
        fold_config['NetModule']['log_dir'] = f"./logs/5-k-validation_3d/{timestamp}/fold_{fold_idx + 1}/"
        
        # Create model
        model = NetModule(config=fold_config)
        
        # Create trainer
        trainer = L.Trainer(
            logger=model.configure_loggers(),
            callbacks=model.configure_callbacks(),
            devices=1 if torch.cuda.is_available() else 0,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            max_epochs=net_config.get('epochs', 500),
            log_every_n_steps=1,
            enable_progress_bar=True
        )
        
        # Setup data module
        datamodule.setup()
        
        # Train the model
        print("Training...")
        trainer.fit(model, datamodule=datamodule)
        
        # Evaluate on validation set
        print("Evaluating...")
        model.to(device)
        y_true, y_pred, y_probs = evaluate_fold(model, datamodule.val_dataloader(), device, num_classes)
        
        # Compute metrics
        results = compute_validation_metrics(y_true, y_pred, y_probs, num_classes)
        all_fold_results.append(results)
        
        # Save fold results
        save_fold_results(results, fold_idx, output_dir)
        
        print(f"Fold {fold_idx + 1} completed - Accuracy: {results['accuracy']:.4f}, F1: {results['f1_macro']:.4f}")
        print("-" * 70)
    
    # Generate summary across all folds
    print("\nGenerating cross-validation summary...")
    
    # Calculate mean and std across folds
    summary_metrics = {}
    metric_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']
    
    for key in metric_keys:
        values = [result[key] for result in all_fold_results]
        summary_metrics[f"{key}_mean"] = np.mean(values)
        summary_metrics[f"{key}_std"] = np.std(values)
    
    # Save summary
    summary_file = Path(output_dir) / "cross_validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("5-FOLD CROSS VALIDATION SUMMARY - 3D AMD GRADING\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {config_path}\n")
        f.write(f"Model type: 3D\n")
        f.write(f"Total samples: {len(label_list)}\n")
        f.write(f"Number of folds: {k_folds}\n\n")
        
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("-" * 40 + "\n")
        for key in metric_keys:
            mean_val = summary_metrics[f"{key}_mean"]
            std_val = summary_metrics[f"{key}_std"]
            f.write(f"{key.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write("\nPER-FOLD RESULTS\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(all_fold_results):
            f.write(f"Fold {i+1}: Acc={result['accuracy']:.4f}, "
                   f"F1={result['f1_macro']:.4f}, AUC={result['auc_macro']:.4f}\n")
    
    # Save summary metrics to CSV
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(Path(output_dir) / "cross_validation_metrics.csv", index=False)
    
    print(f"\n5-fold cross validation completed for 3D model!")
    print(f"Results saved to: {output_dir}")
    print(f"Mean Accuracy: {summary_metrics['accuracy_mean']:.4f} ± {summary_metrics['accuracy_std']:.4f}")
    print(f"Mean F1-Score: {summary_metrics['f1_macro_mean']:.4f} ± {summary_metrics['f1_macro_std']:.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        run_kfold_training(config_path)
    else:
        # Default to 3D config
        run_kfold_training('configs/config_3d.toml')

