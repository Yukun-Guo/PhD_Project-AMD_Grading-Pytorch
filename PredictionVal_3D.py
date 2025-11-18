"""
Comprehensive Validation Script for AMD Grading Model

This script performs comprehensive evaluation on the validation dataset including:
- Loading the best checkpoint
- Running inference on validation data
- Calculating comprehensive metrics (confusion matrix, accuracy, precision, recall, F1, specificity, sensitivity, AUC)
- Saving results to CSV and TXT files
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import lightning as L
import torch.nn.functional as F
from torchmetrics import functional as FM
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, 
    auc, classification_report, multilabel_confusion_matrix
)
import toml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from NetModule_3D import NetModule
from DataModule_3D import DataModel
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def find_latest_checkpoint(checkpoint_dir, model_name):
    """
    Find the latest/best checkpoint file based on validation loss.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        model_name (str): Model name subdirectory
        
    Returns:
        str: Path to the best checkpoint file
    """
    # Handle template variables in paths
    if '${' in checkpoint_dir:
        # If template variables exist, look in the actual logs directory
        logs_path = Path("./logs")
        if not logs_path.exists():
            raise FileNotFoundError(f"Logs directory not found: {logs_path}")
        
        # Search for checkpoint files in any subfolder
        ckpt_files = list(logs_path.rglob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in: {logs_path}")
        
    else:
        # Original logic for resolved paths
        checkpoint_path = Path(checkpoint_dir) / model_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        # Look for checkpoint files
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_path}")
    
    # Find the best checkpoint (lowest validation loss)
    best_ckpt = None
    best_val_loss = float('inf')
    
    print(f"Found {len(ckpt_files)} checkpoint files:")
    for ckpt_file in ckpt_files:
        filename = ckpt_file.name
        print(f"  - {filename}")
        
        if 'val_loss' in filename:
            try:
                # Extract validation loss from filename
                val_loss_str = filename.split('val_loss=')[1].split('.ckpt')[0]
                val_loss = float(val_loss_str)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt_file
            except (IndexError, ValueError):
                continue
    
    if best_ckpt is None:
        # If no validation loss in filename, use the most recent file
        best_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        print("No validation loss found in filenames, using most recent checkpoint")
    
    print(f"Selected checkpoint: {best_ckpt}")
    print(f"Best validation loss: {best_val_loss:.5f}")
    return str(best_ckpt)


def calculate_specificity(y_true, y_pred, num_classes):
    """
    Calculate specificity for each class.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        num_classes (int): Number of classes
        
    Returns:
        dict: Specificity scores per class
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificity_scores = {}
    
    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_scores[f'class_{i}'] = specificity
    
    # Calculate macro average
    specificity_scores['macro'] = np.mean(list(specificity_scores.values()))
    
    return specificity_scores


def convert_to_grouped_labels(y_true, y_pred, y_proba, group_type):
    """
    Convert 4-class labels to grouped classifications.
    
    Args:
        y_true (array): True labels (0, 1, 2, 3)
        y_pred (array): Predicted labels (0, 1, 2, 3)
        y_proba (array): Prediction probabilities for 4 classes
        group_type (str): Type of grouping
            - 'class1_vs_others': Class 0 vs Classes 1,2,3
            - 'early_vs_late': Classes 0,1 vs Classes 2,3
            - 'nonsevere_vs_severe': Classes 0,1,2 vs Class 3
            - 'three_group': Classes 0,1 vs Class 2 vs Class 3
            
    Returns:
        tuple: (grouped_y_true, grouped_y_pred, grouped_y_proba)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred) 
    y_proba = np.array(y_proba)
    
    if group_type == 'class1_vs_others':
        # Class 0 vs Classes 1,2,3
        # Label 0: Class 0, Label 1: Classes 1,2,3
        grouped_y_true = (y_true != 0).astype(int)
        grouped_y_pred = (y_pred != 0).astype(int)
        # For binary: prob of class 0 vs prob of classes 1,2,3
        grouped_y_proba = np.column_stack([
            y_proba[:, 0],  # Class 0 probability
            np.sum(y_proba[:, 1:], axis=1)  # Classes 1,2,3 probability
        ])
        group_names = ['Class_0', 'Classes_1-2-3']
        
    elif group_type == 'early_vs_late':
        # Classes 0,1 vs Classes 2,3
        grouped_y_true = (y_true >= 2).astype(int)
        grouped_y_pred = (y_pred >= 2).astype(int)
        # For binary: prob of classes 0,1 vs prob of classes 2,3
        grouped_y_proba = np.column_stack([
            np.sum(y_proba[:, 0:2], axis=1),  # Classes 0,1 probability
            np.sum(y_proba[:, 2:4], axis=1)   # Classes 2,3 probability
        ])
        group_names = ['Classes_0-1', 'Classes_2-3']
        
    elif group_type == 'nonsevere_vs_severe':
        # Classes 0,1,2 vs Class 3
        grouped_y_true = (y_true == 3).astype(int)
        grouped_y_pred = (y_pred == 3).astype(int)
        # For binary: prob of classes 0,1,2 vs prob of class 3
        grouped_y_proba = np.column_stack([
            np.sum(y_proba[:, 0:3], axis=1),  # Classes 0,1,2 probability
            y_proba[:, 3]  # Class 3 probability
        ])
        group_names = ['Classes_0-1-2', 'Class_3']
        
    elif group_type == 'three_group':
        # Classes 0,1 vs Class 2 vs Class 3
        grouped_y_true = np.where(y_true <= 1, 0, np.where(y_true == 2, 1, 2))
        grouped_y_pred = np.where(y_pred <= 1, 0, np.where(y_pred == 2, 1, 2))
        # For 3-class: prob of classes 0,1 vs class 2 vs class 3
        grouped_y_proba = np.column_stack([
            np.sum(y_proba[:, 0:2], axis=1),  # Classes 0,1 probability
            y_proba[:, 2],  # Class 2 probability
            y_proba[:, 3]   # Class 3 probability
        ])
        group_names = ['Classes_0-1', 'Class_2', 'Class_3']
        
    else:
        raise ValueError(f"Unknown group_type: {group_type}")
    
    return grouped_y_true, grouped_y_pred, grouped_y_proba, group_names


def calculate_grouped_metrics(y_true, y_pred, y_proba):
    """
    Calculate metrics for all grouped classifications.
    
    Args:
        y_true (array): True labels for 4-class classification
        y_pred (array): Predicted labels for 4-class classification  
        y_proba (array): Prediction probabilities for 4-class classification
        
    Returns:
        dict: Dictionary containing metrics for all groupings
    """
    grouped_metrics = {}
    
    # Define all grouping types
    grouping_types = {
        'class1_vs_others': 'Class 0 vs Classes 1,2,3',
        'early_vs_late': 'Classes 0,1 vs Classes 2,3', 
        'nonsevere_vs_severe': 'Classes 0,1,2 vs Class 3',
        'three_group': 'Classes 0,1 vs Class 2 vs Class 3'
    }
    
    for group_type, description in grouping_types.items():
        print(f"Calculating metrics for: {description}")
        
        # Convert labels to grouped format
        g_y_true, g_y_pred, g_y_proba, group_names = convert_to_grouped_labels(
            y_true, y_pred, y_proba, group_type
        )
        
        # Determine number of classes for this grouping
        num_groups = len(group_names)
        
        # Calculate metrics using existing comprehensive function
        group_metrics = calculate_comprehensive_metrics(
            g_y_true, g_y_pred, g_y_proba, num_groups, group_names
        )
        
        # Store with descriptive key
        grouped_metrics[group_type] = {
            'description': description,
            'group_names': group_names,
            'metrics': group_metrics
        }
    
    return grouped_metrics


def calculate_comprehensive_metrics(y_true, y_pred, y_proba, num_classes, class_names=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_proba (array): Prediction probabilities
        num_classes (int): Number of classes
        class_names (list): Optional class names
        
    Returns:
        dict: Comprehensive metrics dictionary
    """
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class and averaged metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Specificity
    specificity_scores = calculate_specificity(y_true, y_pred, num_classes)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # AUC scores (if we have probabilities)
    try:
        if num_classes == 2:
            auc_roc = roc_auc_score(y_true, y_proba[:, 1])
            auc_macro = auc_roc
            auc_micro = auc_roc
            auc_weighted = auc_roc
            auc_per_class = [0.0, auc_roc]  # Class 0 and Class 1
        else:
            auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            auc_micro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
            auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            auc_per_class = roc_auc_score(y_true, y_proba, multi_class='ovr', average=None)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC scores: {e}")
        auc_macro = auc_micro = auc_weighted = 0.0
        auc_per_class = [0.0] * num_classes
    
    # Organize metrics
    metrics = {
        'accuracy': accuracy,
        'precision': {
            'macro': precision_macro,
            'micro': precision_micro,
            'weighted': precision_weighted,
            'per_class': {class_names[i]: precision_per_class[i] for i in range(num_classes)}
        },
        'recall': {
            'macro': recall_macro,
            'micro': recall_micro,
            'weighted': recall_weighted,
            'per_class': {class_names[i]: recall_per_class[i] for i in range(num_classes)}
        },
        'sensitivity': {  # Same as recall
            'macro': recall_macro,
            'micro': recall_micro,
            'weighted': recall_weighted,
            'per_class': {class_names[i]: recall_per_class[i] for i in range(num_classes)}
        },
        'f1_score': {
            'macro': f1_macro,
            'micro': f1_micro,
            'weighted': f1_weighted,
            'per_class': {class_names[i]: f1_per_class[i] for i in range(num_classes)}
        },
        'specificity': specificity_scores,
        'auc_roc': {
            'macro': auc_macro,
            'micro': auc_micro,
            'weighted': auc_weighted,
            'per_class': {class_names[i]: auc_per_class[i] for i in range(num_classes)}
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }
    
    return metrics


def save_metrics_to_csv(metrics, save_path, grouped_metrics=None):
    """
    Save metrics to CSV file in a structured format.
    
    Args:
        metrics (dict): Main 4-class metrics dictionary
        save_path (Path): Path to save CSV file
        grouped_metrics (dict): Optional grouped classification metrics
    """
    rows = []
    
    # Overall accuracy
    rows.append({'Metric': 'Accuracy', 'Type': 'Overall', 'Class': 'All', 'Value': metrics['accuracy']})
    
    # Precision metrics
    for metric_type in ['macro', 'micro', 'weighted']:
        rows.append({
            'Metric': 'Precision', 
            'Type': metric_type.capitalize(), 
            'Class': 'All', 
            'Value': metrics['precision'][metric_type]
        })
    
    for class_name, value in metrics['precision']['per_class'].items():
        rows.append({
            'Metric': 'Precision', 
            'Type': 'Per-Class', 
            'Class': class_name, 
            'Value': value
        })
    
    # Recall/Sensitivity metrics
    for metric_type in ['macro', 'micro', 'weighted']:
        rows.append({
            'Metric': 'Recall', 
            'Type': metric_type.capitalize(), 
            'Class': 'All', 
            'Value': metrics['recall'][metric_type]
        })
    
    for class_name, value in metrics['recall']['per_class'].items():
        rows.append({
            'Metric': 'Recall', 
            'Type': 'Per-Class', 
            'Class': class_name, 
            'Value': value
        })
    
    # F1-Score metrics
    for metric_type in ['macro', 'micro', 'weighted']:
        rows.append({
            'Metric': 'F1-Score', 
            'Type': metric_type.capitalize(), 
            'Class': 'All', 
            'Value': metrics['f1_score'][metric_type]
        })
    
    for class_name, value in metrics['f1_score']['per_class'].items():
        rows.append({
            'Metric': 'F1-Score', 
            'Type': 'Per-Class', 
            'Class': class_name, 
            'Value': value
        })
    
    # Specificity metrics
    for key, value in metrics['specificity'].items():
        if key == 'macro':
            rows.append({
                'Metric': 'Specificity', 
                'Type': 'Macro', 
                'Class': 'All', 
                'Value': value
            })
        else:
            rows.append({
                'Metric': 'Specificity', 
                'Type': 'Per-Class', 
                'Class': key, 
                'Value': value
            })
    
    # AUC metrics
    for metric_type in ['macro', 'micro', 'weighted']:
        rows.append({
            'Metric': 'AUC-ROC', 
            'Type': metric_type.capitalize(), 
            'Class': 'All', 
            'Value': metrics['auc_roc'][metric_type]
        })
    
    for class_name, value in metrics['auc_roc']['per_class'].items():
        rows.append({
            'Metric': 'AUC-ROC', 
            'Type': 'Per-Class', 
            'Class': class_name, 
            'Value': value
        })
    
    # Add grouped metrics if provided
    if grouped_metrics is not None:
        for group_type, group_data in grouped_metrics.items():
            group_desc = group_data['description']
            group_metrics = group_data['metrics']
            
            # Add separator row
            rows.append({
                'Metric': f'=== GROUPED METRICS: {group_desc} ===',
                'Type': '',
                'Class': '',
                'Value': ''
            })
            
            # Accuracy for this grouping
            rows.append({
                'Metric': 'Accuracy',
                'Type': f'Grouped ({group_desc})',
                'Class': 'All',
                'Value': group_metrics['accuracy']
            })
            
            # Precision for this grouping
            for metric_type in ['macro', 'micro', 'weighted']:
                rows.append({
                    'Metric': 'Precision',
                    'Type': f'Grouped-{metric_type.capitalize()} ({group_desc})',
                    'Class': 'All',
                    'Value': group_metrics['precision'][metric_type]
                })
            
            for class_name, value in group_metrics['precision']['per_class'].items():
                rows.append({
                    'Metric': 'Precision',
                    'Type': f'Grouped-Per-Class ({group_desc})',
                    'Class': class_name,
                    'Value': value
                })
            
            # Recall for this grouping
            for metric_type in ['macro', 'micro', 'weighted']:
                rows.append({
                    'Metric': 'Recall',
                    'Type': f'Grouped-{metric_type.capitalize()} ({group_desc})',
                    'Class': 'All',
                    'Value': group_metrics['recall'][metric_type]
                })
            
            for class_name, value in group_metrics['recall']['per_class'].items():
                rows.append({
                    'Metric': 'Recall',
                    'Type': f'Grouped-Per-Class ({group_desc})',
                    'Class': class_name,
                    'Value': value
                })
            
            # F1-Score for this grouping
            for metric_type in ['macro', 'micro', 'weighted']:
                rows.append({
                    'Metric': 'F1-Score',
                    'Type': f'Grouped-{metric_type.capitalize()} ({group_desc})',
                    'Class': 'All',
                    'Value': group_metrics['f1_score'][metric_type]
                })
            
            for class_name, value in group_metrics['f1_score']['per_class'].items():
                rows.append({
                    'Metric': 'F1-Score',
                    'Type': f'Grouped-Per-Class ({group_desc})',
                    'Class': class_name,
                    'Value': value
                })
            
            # AUC-ROC for this grouping
            for metric_type in ['macro', 'micro', 'weighted']:
                rows.append({
                    'Metric': 'AUC-ROC',
                    'Type': f'Grouped-{metric_type.capitalize()} ({group_desc})',
                    'Class': 'All',
                    'Value': group_metrics['auc_roc'][metric_type]
                })
            
            for class_name, value in group_metrics['auc_roc']['per_class'].items():
                rows.append({
                    'Metric': 'AUC-ROC',
                    'Type': f'Grouped-Per-Class ({group_desc})',
                    'Class': class_name,
                    'Value': value
                })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to CSV: {save_path}")


def save_summary_to_txt(metrics, model_info, save_path, grouped_metrics=None):
    """
    Save a comprehensive summary to TXT file.
    
    Args:
        metrics (dict): Main 4-class metrics dictionary
        model_info (dict): Model and experiment information
        save_path (Path): Path to save TXT file
        grouped_metrics (dict): Optional grouped classification metrics
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE VALIDATION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_info.get('model_name', 'Unknown')}\n")
        f.write(f"Checkpoint: {model_info.get('checkpoint_path', 'Unknown')}\n")
        f.write(f"Dataset: Validation Set\n")
        f.write(f"Total Samples: {model_info.get('num_samples', 'Unknown')}\n")
        f.write(f"Number of Classes: {model_info.get('num_classes', 'Unknown')}\n")
        f.write("\n")
        
        # Overall Performance
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['f1_score']['macro']:.4f}\n")
        f.write(f"Micro F1-Score: {metrics['f1_score']['micro']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['f1_score']['weighted']:.4f}\n")
        f.write(f"Macro AUC-ROC: {metrics['auc_roc']['macro']:.4f}\n")
        f.write("\n")
        
        # Per-Class Performance
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12} {'AUC-ROC':<10}\n")
        f.write("-" * 70 + "\n")
        
        for class_name in metrics['precision']['per_class'].keys():
            precision = metrics['precision']['per_class'][class_name]
            recall = metrics['recall']['per_class'][class_name]
            f1 = metrics['f1_score']['per_class'][class_name]
            specificity = metrics['specificity'].get(class_name.lower().replace(' ', '_'), 0.0)
            auc = metrics['auc_roc']['per_class'][class_name]
            
            f.write(f"{class_name:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {specificity:<12.4f} {auc:<10.4f}\n")
        
        f.write("\n")
        
        # Averaged Metrics
        f.write("AVERAGED METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metric':<15} {'Macro':<10} {'Micro':<10} {'Weighted':<10}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Precision':<15} {metrics['precision']['macro']:<10.4f} {metrics['precision']['micro']:<10.4f} {metrics['precision']['weighted']:<10.4f}\n")
        f.write(f"{'Recall':<15} {metrics['recall']['macro']:<10.4f} {metrics['recall']['micro']:<10.4f} {metrics['recall']['weighted']:<10.4f}\n")
        f.write(f"{'F1-Score':<15} {metrics['f1_score']['macro']:<10.4f} {metrics['f1_score']['micro']:<10.4f} {metrics['f1_score']['weighted']:<10.4f}\n")
        f.write(f"{'AUC-ROC':<15} {metrics['auc_roc']['macro']:<10.4f} {metrics['auc_roc']['micro']:<10.4f} {metrics['auc_roc']['weighted']:<10.4f}\n")
        f.write("\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        cm = np.array(metrics['confusion_matrix'])
        
        # Print header
        f.write("True\\Pred ")
        for i in range(cm.shape[1]):
            f.write(f"Class_{i:>8}")
        f.write("\n")
        
        # Print matrix
        for i in range(cm.shape[0]):
            f.write(f"Class_{i:<3} ")
            for j in range(cm.shape[1]):
                f.write(f"{cm[i, j]:>8}")
            f.write("\n")
        
        f.write("\n")
        
        # Detailed Classification Report
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        report = metrics['classification_report']
        
        # Class-wise metrics
        for class_name, class_metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n")
        
        # Summary averages
        f.write(f"\nMacro Average:\n")
        f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-Score: {report['macro avg']['f1-score']:.4f}\n")
        
        f.write(f"\nWeighted Average:\n")
        f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Recall: {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
        
        # Add grouped metrics if provided
        if grouped_metrics is not None:
            f.write("\n" + "=" * 80 + "\n")
            f.write("GROUPED CLASSIFICATION METRICS\n")
            f.write("=" * 80 + "\n")
            
            for group_type, group_data in grouped_metrics.items():
                group_desc = group_data['description']
                group_names = group_data['group_names']
                group_metrics = group_data['metrics']
                
                f.write(f"\n{group_desc.upper()}\n")
                f.write("-" * len(group_desc) + "\n")
                
                # Overall performance for this grouping
                f.write(f"Overall Accuracy: {group_metrics['accuracy']:.4f}\n")
                f.write(f"Macro F1-Score: {group_metrics['f1_score']['macro']:.4f}\n")
                f.write(f"Macro AUC-ROC: {group_metrics['auc_roc']['macro']:.4f}\n\n")
                
                # Per-group performance
                f.write("Per-Group Performance:\n")
                f.write(f"{'Group':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}\n")
                f.write("-" * 65 + "\n")
                
                for group_name in group_names:
                    precision = group_metrics['precision']['per_class'][group_name]
                    recall = group_metrics['recall']['per_class'][group_name]
                    f1 = group_metrics['f1_score']['per_class'][group_name]
                    auc = group_metrics['auc_roc']['per_class'][group_name]
                    
                    f.write(f"{group_name:<20} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auc:<10.4f}\n")
                
                # Confusion Matrix for this grouping
                f.write(f"\nConfusion Matrix ({group_desc}):\n")
                cm = np.array(group_metrics['confusion_matrix'])
                
                # Print header
                f.write("True\\Pred ")
                for i, name in enumerate(group_names):
                    f.write(f"{name:>12}")
                f.write("\n")
                
                # Print matrix
                for i, name in enumerate(group_names):
                    f.write(f"{name:<10}")
                    for j in range(cm.shape[1]):
                        f.write(f"{cm[i, j]:>12}")
                    f.write("\n")
                
                f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Summary report saved to: {save_path}")


def save_confusion_matrix_plot(cm, class_names, save_path):
    """
    Save confusion matrix as a heatmap plot.
    
    Args:
        cm (array): Confusion matrix
        class_names (list): Class names
        save_path (Path): Path to save plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def save_grouped_confusion_matrices(grouped_metrics, output_dir):
    """
    Save confusion matrix plots for all grouped classifications.
    
    Args:
        grouped_metrics (dict): Grouped classification metrics
        output_dir (Path): Directory to save plots
    """
    if grouped_metrics is None:
        return
        
    for group_type, group_data in grouped_metrics.items():
        group_desc = group_data['description']
        group_names = group_data['group_names']
        group_metrics = group_data['metrics']
        cm = np.array(group_metrics['confusion_matrix'])
        
        # Create a filename-safe version of the description
        safe_desc = group_desc.replace(' vs ', '_vs_').replace(' ', '_').replace(',', '').lower()
        plot_path = output_dir / f"confusion_matrix_grouped_{safe_desc}.png"
        
        # Create the plot
        fig_size = (8, 6) if len(group_names) <= 2 else (10, 8)
        plt.figure(figsize=fig_size)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=group_names, yticklabels=group_names)
        plt.title(f'Confusion Matrix: {group_desc}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grouped confusion matrix plot saved to: {plot_path}")


def run_comprehensive_validation():
    """
    Main function to run comprehensive validation evaluation.
    """
    print("Starting comprehensive validation evaluation...")
    
    # Set random seed for reproducibility
    L.seed_everything(1234)

    config_file = "configs/config_3d.toml"
    print(f"Using config file: {config_file}")
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = toml.load(f)
    
    print("Configuration loaded successfully.")
    
    # Initialize data module
    print("Initializing data module...")
    data_model = DataModel(config=config)
    data_model.setup()
    
    # Find and load the latest checkpoint
    print("Loading model checkpoint...")
    model_path = find_latest_checkpoint(
        config['NetModule']['log_dir'], 
        config['NetModule']['model_name']
    )
    
    net_model = NetModule.load_from_checkpoint(model_path, config=config)
    print(f"Model loaded successfully from: {model_path}")
    
    # Set model to evaluation mode
    net_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_model = net_model.to(device)
    print(f"Using device: {device}")
    
    # Get validation dataloader
    val_dataloader = data_model.val_dataloader()
    print(f"Validation dataset size: {len(val_dataloader.dataset)} samples")
    print(f"Number of batches: {len(val_dataloader)}")
    
    # Collect predictions and ground truth
    print("Running inference on validation set...")
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Processing batches")):
            # For 3D model: single volume input [B, C, D, H, W]
            mat, y, _ = batch
            mat, y = mat.to(device), y.to(device)
            
            # Forward pass
            logits = net_model(mat)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(predictions.cpu().numpy())
            all_y_proba.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_proba = np.array(all_y_proba)
    
    print(f"Collected predictions for {len(y_true)} samples")
    print(f"Label distribution - True: {np.bincount(y_true)}")
    print(f"Label distribution - Pred: {np.bincount(y_pred)}")
    
    # Define class names for AMD grading
    num_classes = config['DataModule']['n_class']
    if num_classes == 4:
        # AMD grading classes (customize based on your specific labels)
        class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
    else:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Calculate comprehensive metrics
    print("Calculating comprehensive metrics...")
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba, num_classes, class_names)
    
    # Calculate grouped metrics
    print("Calculating grouped classification metrics...")
    grouped_metrics = calculate_grouped_metrics(y_true, y_pred, y_proba)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("logs") / "validation_results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model information for reporting
    model_info = {
        'model_name': config['NetModule']['model_name'],
        'checkpoint_path': model_path,
        'num_samples': len(y_true),
        'num_classes': num_classes,
        'config_file': str(config_file)
    }
    
    # Save results
    print("Saving results...")
    
    # Save metrics to CSV
    csv_path = output_dir / "validation_metrics.csv"
    save_metrics_to_csv(metrics, csv_path, grouped_metrics)
    
    # Save summary to TXT
    txt_path = output_dir / "validation_summary.txt"
    save_summary_to_txt(metrics, model_info, txt_path, grouped_metrics)
    
    # Save confusion matrix plot
    cm_plot_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix_plot(metrics['confusion_matrix'], class_names, cm_plot_path)
    
    # Save grouped confusion matrix plots
    save_grouped_confusion_matrices(grouped_metrics, output_dir)
    
    # Save raw predictions and probabilities
    results_data = {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_proba.tolist(),
        'metrics': metrics,
        'grouped_metrics': grouped_metrics,
        'model_info': model_info
    }
    
    json_path = output_dir / "validation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nValidation evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey Metrics (4-Class):")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1-Score: {metrics['f1_score']['macro']:.4f}")
    print(f"  Macro AUC-ROC: {metrics['auc_roc']['macro']:.4f}")
    
    print(f"\nGrouped Classification Metrics:")
    for group_type, group_data in grouped_metrics.items():
        group_desc = group_data['description']
        group_metrics = group_data['metrics']
        print(f"  {group_desc}:")
        print(f"    Accuracy: {group_metrics['accuracy']:.4f}")
        print(f"    Macro F1-Score: {group_metrics['f1_score']['macro']:.4f}")
        print(f"    Macro AUC-ROC: {group_metrics['auc_roc']['macro']:.4f}")
    
    return metrics, grouped_metrics, output_dir


if __name__ == "__main__":
    try:
        metrics, grouped_metrics, output_dir = run_comprehensive_validation()
        print("\n" + "="*50)
        print("VALIDATION EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*50)
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        raise
