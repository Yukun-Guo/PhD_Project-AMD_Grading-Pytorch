"""
ModelGradCAM.py - Comprehensive Gradient-Based Visualization Analysis for Entire Dataset

This script performs gradient-based visualization on the entire dataset, supporting multiple methods:
1. Grad-CAM: Class activation mapping using gradients
2. Grad-CAM++: Improved activation mapping with pixel-wise weighting
3. SmoothGrad: Noise-based gradient averaging for smoother visualizations
4. VarGrad: Gradient variance analysis for uncertainty visualization

Features:
- Heatmap generation for all samples in train/validation/test sets
- Class-wise analysis and statistics  
- Organized output structure in CAMVis folder
- Summary reports and visualizations
- Batch processing with progress tracking
- Support for individual methods or comprehensive comparison

Usage:
    python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method gradcam
    python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method gradcam++
    python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method smoothgrad --n_samples 50
    python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method vargrad --noise_level 0.2
    python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method both    # Grad-CAM + Grad-CAM++
    python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method all     # All 4 methods
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import toml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lightning as L

# Import project modules
from NetModule import NetModule
from DataModule import DataModel
from Utils.grad_cam import GradCAMVisualizer, ComparisonVisualizer


class DatasetGradCAMAnalyzer:
    """
    Comprehensive Grad-CAM analyzer for entire datasets.
    
    This class handles:
    - Loading trained models and datasets
    - Batch processing of samples with progress tracking
    - Organized output structure
    - Statistical analysis and reporting
    - Memory management for large datasets
    """
    
    def __init__(self, config_path: str, output_dir: str = "CAMVis", method: str = "gradcam",
                 n_samples: int = 50, noise_level: float = 0.15):
        """
        Initialize the analyzer.
        
        Args:
            config_path: Path to configuration file
            output_dir: Base directory for saving results
            method: Visualization method ('gradcam', 'gradcam++', 'smoothgrad', 'vargrad', 'both', or 'all')
            n_samples: Number of noisy samples for SmoothGrad/VarGrad
            noise_level: Noise standard deviation for SmoothGrad/VarGrad
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.method = method.lower()
        self.n_samples = n_samples
        self.noise_level = noise_level
        
        # Load configuration
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = toml.load(f)
        
        # Initialize components
        self.model = None
        self.data_module = None
        self.visualizer = None
        self.comparison_visualizer = None
        self.class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
        
        # Setup output directories
        self.setup_output_structure()
        
        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'class_distribution': {},
            'prediction_accuracy': {},
            'processing_times': [],
            'heatmap_statistics': {}
        }
    
    def setup_output_structure(self):
        """Create organized output directory structure."""
        print(f"Setting up output structure in: {self.output_dir}")
        
        # Main directories
        self.session_dir = self.output_dir / f"session_{self.timestamp}"
        self.heatmaps_dir = self.session_dir / "heatmaps"
        self.visualizations_dir = self.session_dir / "visualizations"
        self.reports_dir = self.session_dir / "reports"
        self.summaries_dir = self.session_dir / "summaries"
        
        # Create all directories
        for dir_path in [self.session_dir, self.heatmaps_dir, self.visualizations_dir, 
                        self.reports_dir, self.summaries_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different dataset splits
        for split in ['train', 'val', 'test']:
            (self.heatmaps_dir / split).mkdir(exist_ok=True)
            (self.visualizations_dir / split).mkdir(exist_ok=True)
        
        # Class-wise directories
        for i, class_name in enumerate(self.class_names):
            class_dir = class_name.lower().replace(' ', '_')
            for split in ['train', 'val', 'test']:
                (self.heatmaps_dir / split / f"class_{i}_{class_dir}").mkdir(exist_ok=True)
        
        print(f"✓ Output structure created in: {self.session_dir}")
    
    def load_model_and_data(self):
        """Load the trained model and setup data module."""
        print("Loading model and data...")
        
        # Setup data module
        self.data_module = DataModel(config=self.config)
        self.data_module.setup()
        
        # Find and load model checkpoint
        checkpoint_path = self.find_latest_checkpoint(
            self.config['NetModule']['log_dir'], 
            self.config['NetModule']['model_name']
        )
        
        self.model = NetModule.load_from_checkpoint(checkpoint_path, config=self.config)
        self.model.eval()
        
        # Create visualizer(s) based on method
        use_cuda = torch.cuda.is_available()
        
        if self.method == 'both':
            self.comparison_visualizer = ComparisonVisualizer(self.model, use_cuda=use_cuda)
            self.visualizer = None  # Will use comparison_visualizer instead
            print(f"✓ Comparison visualizer ready (Grad-CAM + Grad-CAM++, CUDA: {use_cuda})")
        elif self.method in ['all', 'smoothgrad', 'vargrad']:
            # For these methods, we create visualizers on-demand in analyze_single_sample
            self.visualizer = None
            self.comparison_visualizer = None
            print(f"✓ Method '{self.method}' ready - visualizers will be created on-demand (CUDA: {use_cuda})")
        else:
            # Traditional gradcam or gradcam++ methods
            self.visualizer = GradCAMVisualizer(self.model, use_cuda=use_cuda, method=self.method)
            print(f"✓ {self.method.upper()} visualizer ready (CUDA: {use_cuda})")
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Data module setup complete")
    
    def find_latest_checkpoint(self, checkpoint_dir: str, model_name: str) -> str:
        """Find the latest checkpoint file."""
        checkpoint_path = Path(checkpoint_dir) / model_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_path}")
        
        latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
        return str(latest_ckpt)
    
    def analyze_dataset_split(self, split: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze a specific dataset split (train/val/test).
        
        Args:
            split: Dataset split name ('train', 'val', 'test')
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*50}")
        print(f"ANALYZING {split.upper()} DATASET")
        print(f"{'='*50}")
        
        # Get data loader
        if split == 'train':
            dataloader = self.data_module.train_dataloader()
        elif split == 'val':
            dataloader = self.data_module.val_dataloader()
        elif split == 'test':
            dataloader = self.data_module.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Calculate total samples
        total_samples = len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else 0
        if max_samples:
            total_samples = min(total_samples, max_samples)
        
        print(f"Processing {total_samples} samples from {split} set...")
        
        # Initialize split statistics
        split_stats = {
            'samples_processed': 0,
            'class_counts': {i: 0 for i in range(self.config['DataModule']['n_class'])},
            'predictions': [],
            'ground_truths': [],
            'confidences': [],
            'processing_times': [],
            'heatmap_stats': {inp: [] for inp in ['mnv', 'fluid', 'ga', 'drusen']}
        }
        
        # Process samples
        sample_count = 0
        with tqdm(total=total_samples, desc=f"Processing {split}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                
                mnv, fluid, ga, drusen, targets, sample_ids = batch
                batch_size = mnv.size(0)
                
                # Process each sample in the batch
                for i in range(batch_size):
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    start_time = time.time()
                    
                    # Extract single sample
                    sample_inputs = [
                        mnv[i:i+1], fluid[i:i+1], 
                        ga[i:i+1], drusen[i:i+1]
                    ]
                    true_class = int(targets[i].item())
                    sample_id = sample_ids[i] if isinstance(sample_ids, list) else f"{split}_sample_{sample_count:05d}"
                    
                    # Perform Grad-CAM analysis
                    try:
                        results = self.analyze_single_sample(
                            sample_inputs, true_class, sample_id, split
                        )
                        
                        # Update statistics
                        pred_class = results['predicted_class']
                        confidence = results['confidence']
                        
                        split_stats['samples_processed'] += 1
                        split_stats['class_counts'][true_class] += 1
                        split_stats['predictions'].append(pred_class)
                        split_stats['ground_truths'].append(true_class)
                        split_stats['confidences'].append(confidence)
                        
                        # Heatmap statistics
                        # Handle different heatmap structures based on method
                        if self.method == 'all':
                            # For 'all' method, use gradcam results as the primary statistics
                            # Structure: {'gradcam': {'mnv': array, ...}, 'gradcam++': {...}, ...}
                            primary_method = 'gradcam'  # Use gradcam as representative
                            if primary_method in results['heatmaps']:
                                for inp_name, heatmap in results['heatmaps'][primary_method].items():
                                    if inp_name in split_stats['heatmap_stats']:
                                        max_activation = float(heatmap.max())
                                        mean_activation = float(heatmap.mean())
                                        split_stats['heatmap_stats'][inp_name].append({
                                            'max': max_activation,
                                            'mean': mean_activation,
                                            'sample_id': sample_id
                                        })
                        else:
                            # For single methods, structure is flat: {'mnv': array, 'fluid': array, ...}
                            for inp_name, heatmap in results['heatmaps'].items():
                                if inp_name in split_stats['heatmap_stats']:
                                    max_activation = float(heatmap.max())
                                    mean_activation = float(heatmap.mean())
                                    split_stats['heatmap_stats'][inp_name].append({
                                        'max': max_activation,
                                        'mean': mean_activation,
                                        'sample_id': sample_id
                                    })
                        
                        processing_time = time.time() - start_time
                        split_stats['processing_times'].append(processing_time)
                        
                    except Exception as e:
                        print(f"Error processing {sample_id}: {e}")
                        continue
                    
                    sample_count += 1
                    pbar.update(1)
        
        # Calculate accuracy
        if split_stats['predictions'] and split_stats['ground_truths']:
            correct = sum(p == g for p, g in zip(split_stats['predictions'], split_stats['ground_truths']))
            accuracy = correct / len(split_stats['predictions'])
            split_stats['accuracy'] = accuracy
        else:
            split_stats['accuracy'] = 0.0
        
        print(f"✓ {split} analysis complete: {split_stats['samples_processed']} samples processed")
        print(f"  Accuracy: {split_stats['accuracy']:.1%}")
        print(f"  Avg processing time: {np.mean(split_stats['processing_times']):.2f}s per sample")
        
        return split_stats
    
    def analyze_single_sample(self, inputs: List[torch.Tensor], true_class: int, 
                            sample_id: str, split: str) -> Dict[str, Any]:
        """
        Analyze a single sample and save results.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            true_class: True class label
            sample_id: Sample identifier
            split: Dataset split name
            
        Returns:
            Analysis results dictionary
        """
        from Utils.grad_cam import (AllMethodsComparison, SmoothGradVisualizer, VarGradVisualizer, 
                                  MultiInputGradCAM, MultiInputGradCAMPlusPlus)
        
        # Ensure inputs are on the same device as model
        device = next(self.model.parameters()).device
        inputs = [inp.to(device) for inp in inputs]
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(*inputs)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Generate heatmaps based on method
        if self.method == 'all':
            # Use all methods comparison
            all_methods = AllMethodsComparison(self.model, use_cuda=True, 
                                             n_samples=self.n_samples, noise_level=self.noise_level)
            all_results = all_methods.compare_all_methods(inputs, target_class=predicted_class)
            
            # Save all methods results
            self.save_all_methods_results(inputs, all_results, {
                'sample_id': sample_id,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0]
            }, split)
            
            all_methods.cleanup()
            heatmaps = all_results
            
        elif self.method == 'both':
            # Use comparison visualizer (Grad-CAM vs Grad-CAM++)
            grad_cam_heatmaps, grad_cam_pp_heatmaps = self.comparison_visualizer.compare_methods(
                inputs, target_class=predicted_class
            )
            
            # Save comparison results
            self.save_comparison_results(inputs, grad_cam_heatmaps, grad_cam_pp_heatmaps, {
                'sample_id': sample_id,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0]
            }, split)
            
            heatmaps = {'gradcam': grad_cam_heatmaps, 'gradcam++': grad_cam_pp_heatmaps}
            
        elif self.method == 'smoothgrad':
            # Use SmoothGrad
            smooth_grad = SmoothGradVisualizer(self.model, use_cuda=True, 
                                             n_samples=self.n_samples, noise_level=self.noise_level)
            heatmaps = smooth_grad.analyze_sample(inputs, target_class=predicted_class)
            
            # Save results
            self.save_single_method_results(inputs, heatmaps, {
                'sample_id': sample_id,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0],
                'method': 'smoothgrad'
            }, split)
            
            smooth_grad.cleanup()
            
        elif self.method == 'vargrad':
            # Use VarGrad
            var_grad = VarGradVisualizer(self.model, use_cuda=True, 
                                       n_samples=self.n_samples, noise_level=self.noise_level)
            heatmaps = var_grad.analyze_sample(inputs, target_class=predicted_class)
            
            # Save results
            self.save_single_method_results(inputs, heatmaps, {
                'sample_id': sample_id,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0],
                'method': 'vargrad'
            }, split)
            
            var_grad.cleanup()
            
        elif self.method in ['gradcam', 'gradcam++']:
            # Use single method visualizer
            heatmaps = self.visualizer.grad_cam.generate_cam(inputs, target_class=predicted_class)
            
            # Save results
            self.save_sample_results(inputs, heatmaps, {
                'sample_id': sample_id,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities.cpu().numpy()[0]
            }, split)
        
        return {
            'sample_id': sample_id,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'heatmaps': heatmaps,
            'method': self.method
        }
    
    def save_sample_results(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray],
                          info: Dict[str, Any], split: str):
        """Save individual sample results."""
        sample_id = info['sample_id']
        true_class = info['true_class']
        pred_class = info['predicted_class']
        
        # Clean sample_id to get just the filename without path
        if isinstance(sample_id, str) and ('/' in sample_id or '\\' in sample_id):
            sample_id = Path(sample_id).stem  # Get filename without extension
        
        # Determine class directory
        class_name = self.class_names[true_class].lower().replace(' ', '_')
        class_dir = self.heatmaps_dir / split / f"class_{true_class}_{class_name}"
        
        # Save individual heatmaps
        heatmap_dir = class_dir / sample_id
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary visualizer if needed
        if self.visualizer is None:
            from Utils.grad_cam import GradCAMVisualizer
            temp_visualizer = GradCAMVisualizer(self.model, use_cuda=torch.cuda.is_available(), method=self.method)
            temp_visualizer.grad_cam.save_heatmaps(heatmaps, heatmap_dir, sample_id)
            temp_visualizer.cleanup()
        else:
            self.visualizer.grad_cam.save_heatmaps(heatmaps, heatmap_dir, sample_id)
        
        # Save combined visualization
        viz_dir = self.visualizations_dir / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f"{sample_id}_combined.png"
        
        # Create a temporary visualizer for visualization if needed
        if self.visualizer is None:
            from Utils.grad_cam import GradCAMVisualizer
            temp_visualizer = GradCAMVisualizer(self.model, use_cuda=torch.cuda.is_available(), method=self.method)
            temp_visualizer.grad_cam.visualize_cam(inputs, heatmaps, viz_path, pred_class)
            temp_visualizer.cleanup()
        else:
            self.visualizer.grad_cam.visualize_cam(inputs, heatmaps, viz_path, pred_class)
        
        # Save sample metadata
        metadata = {
            'sample_id': sample_id,
            'true_class': true_class,
            'true_class_name': self.class_names[true_class],
            'predicted_class': pred_class,
            'predicted_class_name': self.class_names[pred_class],
            'confidence': info['confidence'],
            'probabilities': info['probabilities'].tolist(),
            'correct_prediction': true_class == pred_class,
            'heatmap_statistics': {
                inp_name: {
                    'max_activation': float(heatmap.max()),
                    'mean_activation': float(heatmap.mean()),
                    'std_activation': float(heatmap.std())
                }
                for inp_name, heatmap in heatmaps.items()
            }
        }
        
        metadata_path = heatmap_dir / f"{sample_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_comparison_results(self, inputs: List[torch.Tensor], 
                              grad_cam_heatmaps: Dict[str, np.ndarray],
                              grad_cam_pp_heatmaps: Dict[str, np.ndarray],
                              info: Dict[str, Any], split: str):
        """Save comparison results for both Grad-CAM and Grad-CAM++."""
        sample_id = info['sample_id']
        true_class = info['true_class']
        pred_class = info['predicted_class']
        
        # Clean sample_id to get just the filename without path
        if isinstance(sample_id, str) and ('/' in sample_id or '\\' in sample_id):
            sample_id = Path(sample_id).stem  # Get filename without extension
        
        # Determine class directory
        class_name = self.class_names[true_class].lower().replace(' ', '_')
        class_dir = self.heatmaps_dir / split / f"class_{true_class}_{class_name}"
        
        # Save individual heatmaps for both methods
        gradcam_dir = class_dir / f"{sample_id}_gradcam"
        gradcampp_dir = class_dir / f"{sample_id}_gradcampp"
        
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        gradcampp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save heatmaps using the comparison visualizer's methods
        self.comparison_visualizer.grad_cam.save_heatmaps(grad_cam_heatmaps, gradcam_dir, f"{sample_id}_gradcam")
        self.comparison_visualizer.grad_cam_pp.save_heatmaps(grad_cam_pp_heatmaps, gradcampp_dir, f"{sample_id}_gradcampp")
        
        # Save comparison visualization
        viz_dir = self.visualizations_dir / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = viz_dir / f"{sample_id}_comparison.png"
        
        self.comparison_visualizer._create_comparison_plot(
            inputs, grad_cam_heatmaps, grad_cam_pp_heatmaps, pred_class, comparison_path
        )
        
        # Save comparison metadata
        metadata = {
            'sample_id': sample_id,
            'true_class': true_class,
            'true_class_name': self.class_names[true_class],
            'predicted_class': pred_class,
            'predicted_class_name': self.class_names[pred_class],
            'confidence': info['confidence'],
            'probabilities': info['probabilities'].tolist(),
            'correct_prediction': true_class == pred_class,
            'method': 'comparison',
            'gradcam_statistics': {
                inp_name: {
                    'max_activation': float(heatmap.max()),
                    'mean_activation': float(heatmap.mean()),
                    'std_activation': float(heatmap.std())
                }
                for inp_name, heatmap in grad_cam_heatmaps.items()
            },
            'gradcampp_statistics': {
                inp_name: {
                    'max_activation': float(heatmap.max()),
                    'mean_activation': float(heatmap.mean()),
                    'std_activation': float(heatmap.std())
                }
                for inp_name, heatmap in grad_cam_pp_heatmaps.items()
            }
        }
        
        metadata_path = class_dir / f"{sample_id}_comparison_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_single_method_results(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray],
                                  info: Dict[str, Any], split: str):
        """Save individual sample results for single method (SmoothGrad/VarGrad)."""
        sample_id = info['sample_id']
        true_class = info['true_class']
        pred_class = info['predicted_class']
        method = info['method']
        
        # Clean sample_id to get just the filename without path
        if isinstance(sample_id, str) and ('/' in sample_id or '\\' in sample_id):
            sample_id = Path(sample_id).stem  # Get filename without extension
        
        # Determine class directory
        class_name = self.class_names[true_class].lower().replace(' ', '_')
        class_dir = self.heatmaps_dir / split / f"class_{true_class}_{class_name}"
        
        # Save individual heatmaps
        heatmap_dir = class_dir / f"{sample_id}_{method}"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Save heatmaps using the appropriate method
        for inp_name, heatmap in heatmaps.items():
            filename = f"{sample_id}_{method}_{inp_name}.png"
            filepath = heatmap_dir / filename
            
            # Convert to colored heatmap
            import matplotlib.pyplot as plt
            from matplotlib import cm
            if method == 'vargrad':
                colored_heatmap = plt.cm.get_cmap('plasma')(heatmap)
            else:  # smoothgrad
                colored_heatmap = plt.cm.get_cmap('hot')(heatmap)
            
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Save using PIL
            try:
                from PIL import Image
                heatmap_pil = Image.fromarray(colored_heatmap)
                heatmap_pil.save(str(filepath))
            except ImportError:
                # Fallback to matplotlib
                plt.figure(figsize=(8, 8))
                if method == 'vargrad':
                    plt.imshow(heatmap, cmap='plasma')
                else:
                    plt.imshow(heatmap, cmap='hot')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(filepath, bbox_inches='tight', dpi=150)
                plt.close()
        
        # Create overlay visualization in visualizations directory
        viz_dir = self.visualizations_dir / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f"{sample_id}_{method}_overlay.png"
        
        # Create overlay visualization using the appropriate method
        if method == 'smoothgrad':
            from Utils.grad_cam import MultiInputSmoothGrad
            temp_smooth_grad = MultiInputSmoothGrad(self.model, use_cuda=True, n_samples=self.n_samples, noise_level=self.noise_level)
            temp_smooth_grad.visualize_smooth_grad(inputs, heatmaps, viz_path, pred_class)
        elif method == 'vargrad':
            from Utils.grad_cam import MultiInputVarGrad
            temp_var_grad = MultiInputVarGrad(self.model, use_cuda=True, n_samples=self.n_samples, noise_level=self.noise_level)
            temp_var_grad.visualize_var_grad(inputs, heatmaps, viz_path, pred_class)
        
        # Save sample metadata
        metadata = {
            'sample_id': sample_id,
            'true_class': true_class,
            'true_class_name': self.class_names[true_class],
            'predicted_class': pred_class,
            'predicted_class_name': self.class_names[pred_class],
            'confidence': info['confidence'],
            'probabilities': info['probabilities'].tolist(),
            'correct_prediction': true_class == pred_class,
            'method': method,
            'heatmap_statistics': {
                inp_name: {
                    'max_activation': float(heatmap.max()),
                    'mean_activation': float(heatmap.mean()),
                    'std_activation': float(heatmap.std())
                }
                for inp_name, heatmap in heatmaps.items()
            }
        }
        
        metadata_path = class_dir / f"{sample_id}_{method}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_all_methods_results(self, inputs: List[torch.Tensor], all_results: Dict[str, Dict[str, np.ndarray]],
                                info: Dict[str, Any], split: str):
        """Save results for all methods comparison."""
        sample_id = info['sample_id']
        true_class = info['true_class']
        pred_class = info['predicted_class']
        
        # Clean sample_id to get just the filename without path
        if isinstance(sample_id, str) and ('/' in sample_id or '\\' in sample_id):
            sample_id = Path(sample_id).stem  # Get filename without extension
        
        # Determine class directory
        class_name = self.class_names[true_class].lower().replace(' ', '_')
        class_dir = self.heatmaps_dir / split / f"class_{true_class}_{class_name}"
        
        # Save individual heatmaps for each method
        for method_name, heatmaps in all_results.items():
            method_dir = class_dir / f"{sample_id}_{method_name}"
            method_dir.mkdir(parents=True, exist_ok=True)
            
            # Choose appropriate colormap
            if method_name == 'vargrad':
                colormap = 'plasma'
            elif method_name == 'smoothgrad':
                colormap = 'hot'
            else:  # gradcam, gradcam++
                colormap = 'jet'
            
            # Save heatmaps
            for inp_name, heatmap in heatmaps.items():
                filename = f"{sample_id}_{method_name}_{inp_name}.png"
                filepath = method_dir / filename
                
                # Convert to colored heatmap
                import matplotlib.pyplot as plt
                colored_heatmap = plt.cm.get_cmap(colormap)(heatmap)
                colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                
                # Save using PIL
                try:
                    from PIL import Image
                    heatmap_pil = Image.fromarray(colored_heatmap)
                    heatmap_pil.save(str(filepath))
                except ImportError:
                    # Fallback to matplotlib
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 8))
                    plt.imshow(heatmap, cmap=colormap)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(filepath, bbox_inches='tight', dpi=150)
                    plt.close()
        
        # Save comprehensive comparison visualization
        viz_dir = self.visualizations_dir / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f"{sample_id}_all_methods_comparison.png"
        
        # Create comprehensive overlay visualization using AllMethodsComparison
        from Utils.grad_cam import AllMethodsComparison
        temp_all_methods = AllMethodsComparison(self.model, use_cuda=True, n_samples=self.n_samples, noise_level=self.noise_level)
        temp_all_methods._create_all_methods_plot(inputs, all_results, pred_class, viz_path)
        temp_all_methods.cleanup()
        
        # Save comprehensive metadata
        metadata = {
            'sample_id': sample_id,
            'true_class': true_class,
            'true_class_name': self.class_names[true_class],
            'predicted_class': pred_class,
            'predicted_class_name': self.class_names[pred_class],
            'confidence': info['confidence'],
            'probabilities': info['probabilities'].tolist(),
            'correct_prediction': true_class == pred_class,
            'method': 'all_methods',
            'method_statistics': {}
        }
        
        # Add statistics for each method
        for method_name, heatmaps in all_results.items():
            metadata['method_statistics'][method_name] = {
                inp_name: {
                    'max_activation': float(heatmap.max()),
                    'mean_activation': float(heatmap.mean()),
                    'std_activation': float(heatmap.std())
                }
                for inp_name, heatmap in heatmaps.items()
            }
        
        metadata_path = class_dir / f"{sample_id}_all_methods_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_summary_reports(self, split_results: Dict[str, Dict[str, Any]]):
        """Generate comprehensive summary reports."""
        print("\nGenerating summary reports...")
        
        # Overall statistics
        overall_stats = self.calculate_overall_statistics(split_results)
        
        # Save overall statistics
        stats_path = self.reports_dir / "overall_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(overall_stats, f, indent=2, default=str)
        
        # Generate visualizations
        self.create_summary_visualizations(split_results, overall_stats)
        
        # Create detailed report
        self.create_detailed_report(split_results, overall_stats)
        
        print(f"✓ Summary reports saved to: {self.reports_dir}")
    
    def calculate_overall_statistics(self, split_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics across all splits."""
        stats = {
            'analysis_timestamp': self.timestamp,
            'config_used': self.config_path,
            'model_info': {
                'n_classes': self.config['DataModule']['n_class'],
                'class_names': self.class_names,
                'backbone': self.config['NetModule'].get('backbone_name', 'efficientnet_b5')
            },
            'splits': {}
        }
        
        total_samples = 0
        total_correct = 0
        
        for split_name, split_data in split_results.items():
            split_stats = {
                'samples_processed': split_data['samples_processed'],
                'accuracy': split_data['accuracy'],
                'class_distribution': split_data['class_counts'],
                'avg_processing_time': np.mean(split_data['processing_times']) if split_data['processing_times'] else 0,
                'avg_confidence': np.mean(split_data['confidences']) if split_data['confidences'] else 0
            }
            
            # Heatmap statistics
            heatmap_summary = {}
            for inp_name, heatmap_data in split_data['heatmap_stats'].items():
                if heatmap_data:
                    max_activations = [h['max'] for h in heatmap_data]
                    mean_activations = [h['mean'] for h in heatmap_data]
                    heatmap_summary[inp_name] = {
                        'avg_max_activation': np.mean(max_activations),
                        'std_max_activation': np.std(max_activations),
                        'avg_mean_activation': np.mean(mean_activations),
                        'std_mean_activation': np.std(mean_activations)
                    }
            
            split_stats['heatmap_summary'] = heatmap_summary
            stats['splits'][split_name] = split_stats
            
            total_samples += split_data['samples_processed']
            total_correct += int(split_data['accuracy'] * split_data['samples_processed'])
        
        stats['overall'] = {
            'total_samples_processed': total_samples,
            'overall_accuracy': total_correct / total_samples if total_samples > 0 else 0,
        }
        
        return stats
    
    def create_summary_visualizations(self, split_results: Dict[str, Dict[str, Any]], overall_stats: Dict[str, Any]):
        """Create comprehensive summary visualizations."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy comparison across splits
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy bar plot
        splits = list(split_results.keys())
        accuracies = [split_results[split]['accuracy'] for split in splits]
        
        axes[0, 0].bar(splits, accuracies, color=['skyblue', 'lightgreen', 'salmon'][:len(splits)])
        axes[0, 0].set_title('Accuracy by Dataset Split')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.1%}', ha='center')
        
        # Class distribution
        if split_results:
            split_name = list(split_results.keys())[0]  # Use first split for class distribution
            class_counts = split_results[split_name]['class_counts']
            
            # Only create pie chart if we have data
            if any(count > 0 for count in class_counts.values()):
                axes[0, 1].pie(class_counts.values(), labels=self.class_names, autopct='%1.1f%%')
                axes[0, 1].set_title(f'Class Distribution ({split_name})')
            else:
                axes[0, 1].text(0.5, 0.5, 'No data processed', ha='center', va='center', 
                               transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Class Distribution - No Data')
        else:
            axes[0, 1].text(0.5, 0.5, 'No splits analyzed', ha='center', va='center', 
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Class Distribution - No Data')
        
        # Heatmap activation comparison
        input_names = ['mnv', 'fluid', 'ga', 'drusen']
        avg_activations = {}
        
        for split in splits:
            avg_activations[split] = []
            for inp_name in input_names:
                heatmap_data = split_results[split]['heatmap_stats'].get(inp_name, [])
                if heatmap_data:
                    avg_max = np.mean([h['max'] for h in heatmap_data])
                    avg_activations[split].append(avg_max)
                else:
                    avg_activations[split].append(0)
        
        x = np.arange(len(input_names))
        width = 0.25
        
        for i, split in enumerate(splits):
            axes[1, 0].bar(x + i*width, avg_activations[split], width, label=split)
        
        axes[1, 0].set_xlabel('Input Type')
        axes[1, 0].set_ylabel('Average Max Activation')
        axes[1, 0].set_title('Average Heatmap Activations by Input Type')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels([inp.upper() for inp in input_names])
        axes[1, 0].legend()
        
        # Processing time comparison
        proc_times = [np.mean(split_results[split]['processing_times']) if split_results[split]['processing_times'] else 0 
                     for split in splits]
        
        axes[1, 1].bar(splits, proc_times, color=['orange', 'purple', 'brown'][:len(splits)])
        axes[1, 1].set_title('Average Processing Time by Split')
        axes[1, 1].set_ylabel('Time (seconds)')
        for i, time in enumerate(proc_times):
            axes[1, 1].text(i, time + 0.01, f'{time:.2f}s', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.summaries_dir / 'analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed heatmap analysis
        self.create_heatmap_analysis_plots(split_results)
        
        print("✓ Summary visualizations created")
    
    def create_heatmap_analysis_plots(self, split_results: Dict[str, Dict[str, Any]]):
        """Create detailed heatmap analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Heatmap Activation Analysis', fontsize=16)
        
        input_names = ['mnv', 'fluid', 'ga', 'drusen']
        colors = ['blue', 'green', 'red', 'orange']
        
        for idx, inp_name in enumerate(input_names):
            row, col = idx // 2, idx % 2
            
            # Collect activation data across all splits
            all_max_activations = []
            all_mean_activations = []
            split_labels = []
            
            for split_name, split_data in split_results.items():
                heatmap_data = split_data['heatmap_stats'].get(inp_name, [])
                if heatmap_data:
                    max_acts = [h['max'] for h in heatmap_data]
                    mean_acts = [h['mean'] for h in heatmap_data]
                    
                    all_max_activations.extend(max_acts)
                    all_mean_activations.extend(mean_acts)
                    split_labels.extend([split_name] * len(max_acts))
            
            if all_max_activations:
                # Scatter plot of max vs mean activations
                # Create scatter plot with proper colors for each split
                unique_splits = list(set(split_labels))
                for i, split_name in enumerate(unique_splits):
                    split_indices = [j for j, label in enumerate(split_labels) if label == split_name]
                    if split_indices:
                        split_max = [all_max_activations[j] for j in split_indices]
                        split_mean = [all_mean_activations[j] for j in split_indices]
                        axes[row, col].scatter(split_mean, split_max, 
                                             c=colors[i % len(colors)], alpha=0.6, s=20, 
                                             label=split_name)
                
                axes[row, col].set_xlabel('Mean Activation')
                axes[row, col].set_ylabel('Max Activation')
                axes[row, col].set_title(f'{inp_name.upper()} Heatmap Activations')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend()
            else:
                axes[row, col].text(0.5, 0.5, 'No data available', 
                                  transform=axes[row, col].transAxes, ha='center', va='center')
                axes[row, col].set_title(f'{inp_name.upper()} - No Data')
        
        plt.tight_layout()
        plt.savefig(self.summaries_dir / 'heatmap_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_report(self, split_results: Dict[str, Dict[str, Any]], overall_stats: Dict[str, Any]):
        """Create a detailed text report."""
        report_path = self.reports_dir / 'detailed_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE GRAD-CAM ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config_path}\n")
            f.write(f"Output Directory: {self.session_dir}\n\n")
            
            # Overall Summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Samples Processed: {overall_stats['overall']['total_samples_processed']}\n")
            f.write(f"Overall Accuracy: {overall_stats['overall']['overall_accuracy']:.1%}\n")
            f.write(f"Model Classes: {overall_stats['model_info']['n_classes']}\n")
            f.write(f"Backbone: {overall_stats['model_info']['backbone']}\n\n")
            
            # Split-wise Analysis
            for split_name, split_data in split_results.items():
                f.write(f"{split_name.upper()} DATASET ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Samples Processed: {split_data['samples_processed']}\n")
                f.write(f"Accuracy: {split_data['accuracy']:.1%}\n")
                f.write(f"Average Confidence: {np.mean(split_data['confidences']):.1%}\n")
                f.write(f"Average Processing Time: {np.mean(split_data['processing_times']):.2f}s\n")
                
                f.write("\nClass Distribution:\n")
                for class_idx, count in split_data['class_counts'].items():
                    class_name = self.class_names[class_idx]
                    percentage = count / split_data['samples_processed'] * 100 if split_data['samples_processed'] > 0 else 0
                    f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
                
                f.write("\nHeatmap Statistics:\n")
                for inp_name, heatmap_data in split_data['heatmap_stats'].items():
                    if heatmap_data:
                        max_acts = [h['max'] for h in heatmap_data]
                        mean_acts = [h['mean'] for h in heatmap_data]
                        f.write(f"  {inp_name.upper()}:\n")
                        f.write(f"    Avg Max Activation: {np.mean(max_acts):.4f} ± {np.std(max_acts):.4f}\n")
                        f.write(f"    Avg Mean Activation: {np.mean(mean_acts):.4f} ± {np.std(mean_acts):.4f}\n")
                
                f.write("\n" + "="*50 + "\n")
            
            # Recommendations
            f.write("\nRECOMMENDations & INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            if overall_stats['overall']['overall_accuracy'] > 0.8:
                f.write("✓ Model shows good overall performance (>80% accuracy)\n")
            elif overall_stats['overall']['overall_accuracy'] > 0.6:
                f.write("• Model shows moderate performance (60-80% accuracy)\n")
                f.write("  Consider additional training or data augmentation\n")
            else:
                f.write("⚠ Model shows low performance (<60% accuracy)\n")
                f.write("  Review model architecture, training process, or data quality\n")
            
            f.write("\nHeatmap Analysis Insights:\n")
            # Add specific insights based on heatmap patterns
            f.write("• Check visualizations for clinically relevant activation patterns\n")
            f.write("• Look for consistent activation across similar cases\n")
            f.write("• Verify that model focuses on pathologically relevant regions\n")
            
        print(f"✓ Detailed report saved to: {report_path}")
    
    def run_complete_analysis(self, splits: Optional[List[str]] = None, max_samples_per_split: Optional[int] = None):
        """
        Run complete Grad-CAM analysis on specified dataset splits.
        
        Args:
            splits: List of splits to analyze ['train', 'val', 'test']. If None, analyzes all available.
            max_samples_per_split: Maximum samples per split (None for all)
        """
        print("="*80)
        print("STARTING COMPREHENSIVE DATASET GRAD-CAM ANALYSIS")
        print("="*80)
        
        start_time = time.time()
        
        # Load model and data
        self.load_model_and_data()
        
        # Determine splits to analyze
        if splits is None:
            available_splits = []
            if hasattr(self.data_module, 'train_dataloader') and self.data_module.train_dataloader():
                available_splits.append('train')
            if hasattr(self.data_module, 'val_dataloader') and self.data_module.val_dataloader():
                available_splits.append('val')
            if hasattr(self.data_module, 'test_dataloader') and self.data_module.test_dataloader():
                available_splits.append('test')
            splits = available_splits
        
        print(f"Analyzing splits: {splits}")
        if max_samples_per_split:
            print(f"Max samples per split: {max_samples_per_split}")
        
        # Analyze each split
        split_results = {}
        for split in splits:
            try:
                split_results[split] = self.analyze_dataset_split(split, max_samples_per_split)
            except Exception as e:
                print(f"Error analyzing {split} split: {e}")
                continue
        
        # Generate reports
        if split_results:
            self.generate_summary_reports(split_results)
        
        # Cleanup
        if self.visualizer:
            self.visualizer.cleanup()
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Results saved to: {self.session_dir}")
        print(f"Total samples processed: {sum(r['samples_processed'] for r in split_results.values())}")
        
        # Print quick summary
        print(f"\nQuick Summary:")
        for split_name, split_data in split_results.items():
            print(f"  {split_name}: {split_data['samples_processed']} samples, {split_data['accuracy']:.1%} accuracy")
        
        return split_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Comprehensive Grad-CAM Analysis for Entire Dataset')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test', 'all'], default='val',
                       help='Dataset split(s) to analyze')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples per split (None for all)')
    parser.add_argument('--output_dir', type=str, default='CAMVis',
                       help='Output directory name')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--method', type=str, 
                       choices=['gradcam', 'gradcam++', 'smoothgrad', 'vargrad', 'both', 'all'], 
                       default='gradcam',
                       help='Visualization method: gradcam, gradcam++, smoothgrad, vargrad, both (gradcam+gradcam++), or all (all methods)')
    parser.add_argument('--n_samples', type=int, default=50,
                       help='Number of noisy samples for SmoothGrad/VarGrad (default: 50)')
    parser.add_argument('--noise_level', type=float, default=0.15,
                       help='Noise level (std) for SmoothGrad/VarGrad (default: 0.15)')
    
    args = parser.parse_args()
    
    # Setup splits to analyze
    if args.dataset == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.dataset]
    
    try:
        # Create analyzer
        analyzer = DatasetGradCAMAnalyzer(args.config, args.output_dir, args.method, 
                                         args.n_samples, args.noise_level)
        
        # Run analysis
        results = analyzer.run_complete_analysis(splits, args.max_samples)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Check results in: {analyzer.session_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
    

# Example Usage Commands:

# # Individual methods
# python ModelGradCAM.py --config configs/config_bio.toml --dataset all --method gradcam
# python ModelGradCAM.py --config configs/config_bio.toml --dataset all --method gradcam++
# python ModelGradCAM.py --config configs/config_bio.toml --dataset all --method smoothgrad --n_samples 25
# python ModelGradCAM.py --config configs/config_bio.toml --dataset all --method vargrad --noise_level 0.2

# # Comparison modes
# python ModelGradCAM.py --config configs/config_bio.toml --dataset all --method both
# python ModelGradCAM.py --config configs/config_bio.toml --dataset all --method all  # ✅ THIS NOW WORKS!

# # The original failing command now works:
# python ModelGradCAM.py --config configs/config_oct.toml --dataset all --method all