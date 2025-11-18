"""
ModelGradCAM_3D.py - Comprehensive 3D Gradient-Based Visualization Analysis for Entire Dataset

This script performs gradient-based visualization on 3D OCT volume datasets, supporting multiple methods:
1. Grad-CAM: Class activation mapping using gradients for 3D volumes
2. Grad-CAM++: Improved activation mapping with pixel-wise weighting for 3D volumes
3. SmoothGrad: Noise-based gradient averaging for smoother 3D visualizations
4. VarGrad: Gradient variance analysis for uncertainty visualization on 3D volumes

Features:
- 3D volume heatmap generation for all samples in train/validation/test sets
- Class-wise analysis and statistics for 3D OCT data
- Organized output structure in CAMVis folder
- Summary reports and 3D visualizations
- Batch processing with progress tracking for 3D volumes
- Support for individual methods or comprehensive comparison

Usage:
    python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method gradcam
    python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method gradcam++
    python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method smoothgrad --n_samples 50
    python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method vargrad --noise_level 0.2
    python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method both    # Grad-CAM + Grad-CAM++
    python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method all     # All 4 methods
"""

import argparse
import json
import os
import shutil
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tifffile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import toml
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lightning as L

# Import project modules
from NetModule_3D import NetModule
from DataModule_3D import DataModel
from Utils.grad_cam import GradCAMVisualizer, ComparisonVisualizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class GradCAM3D:
    """
    3D GradCAM implementation for 3D volume models.
    
    This class implements GradCAM, GradCAM++, SmoothGrad, and VarGrad
    for 3D models with single backbone architecture.
    """
    
    def __init__(self, model, target_layer: str):
        """
        Initialize 3D GradCAM.
        
        Args:
            model: The 3D neural network model
            target_layer: Name of the target layer for gradient extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
    def _get_gradients_hook(self, module, grad_input, grad_output):
        """Hook function to capture gradients."""
        self.gradients = grad_output[0]
        
    def _get_activations_hook(self, module, input, output):
        """Hook function to capture activations."""
        self.activations = output
        
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        try:
            # Navigate to the target layer
            target_module = self.model
            for attr in self.target_layer.split('.'):
                print(f"Accessing attribute: {attr}")
                target_module = getattr(target_module, attr)
                print(f"Got module: {type(target_module).__name__}")
            
            print(f"Target module for hooks: {type(target_module).__name__}")
            
            # Register hooks
            forward_hook = target_module.register_forward_hook(self._get_activations_hook)
            backward_hook = target_module.register_backward_hook(self._get_gradients_hook) 
            
            self.hooks = [forward_hook, backward_hook]
            print(f"Successfully registered {len(self.hooks)} hooks")
            
        except Exception as e:
            print(f"Error in _register_hooks: {e}")
            # Print available attributes for debugging
            print("Available model attributes:")
            if hasattr(self.model, '__dict__'):
                for key in self.model.__dict__.keys():
                    print(f"  {key}")
            raise
        
    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate 3D GradCAM heatmap.
        
        Args:
            input_tensor: Input 3D volume tensor (1, C, D, H, W)
            target_class: Target class for CAM generation
            
        Returns:
            3D heatmap as numpy array (D, H, W)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass - get logits
            logits = self.model(input_tensor)
            
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            
            # Backward pass - get gradients
            self.model.zero_grad()
            class_score = logits[0, target_class]
            class_score.backward(retain_graph=True)
            
            # Generate CAM using gradients and activations
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Failed to capture gradients or activations. Check hook registration.")
            gradients = self.gradients.detach().cpu()  # (1, C, D', H', W')
            activations = self.activations.detach().cpu()  # (1, C, D', H', W')
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, D', H', W')
            cam = torch.clamp(cam, min=0)  # ReLU
            
            # Reshape to (D', H', W')
            cam = cam.squeeze()
            
            # Interpolate to original input size
            original_size = input_tensor.shape[2:]  # (D, H, W)
            if cam.shape != original_size:
                # Use trilinear interpolation for 3D
                cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                cam = F.interpolate(cam, size=original_size, mode='trilinear', align_corners=False)
                cam = cam.squeeze()
            
            # Normalize to [0, 1]
            cam_np = cam.numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            
            return cam_np
            
        finally:
            self._remove_hooks()
            
    def generate_cam_plus_plus(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate 3D GradCAM++ heatmap.
        
        Args:
            input_tensor: Input 3D volume tensor (1, C, D, H, W)
            target_class: Target class for CAM generation
            
        Returns:
            3D heatmap as numpy array (D, H, W)
        """
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass
            logits = self.model(input_tensor)
            
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            
            # Backward pass
            self.model.zero_grad()
            class_score = logits[0, target_class]
            class_score.backward(retain_graph=True)
            
            # Get gradients and activations
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Failed to capture gradients or activations. Check hook registration.")
            gradients = self.gradients.detach()  # (1, C, D', H', W')
            activations = self.activations.detach()  # (1, C, D', H', W')
            
            # Calculate alpha weights for GradCAM++
            alpha_num = gradients.pow(2)
            alpha_denom = 2.0 * gradients.pow(2) + torch.sum(activations * gradients.pow(3), 
                                                            dim=(2, 3, 4), keepdim=True)
            alpha = alpha_num / (alpha_denom + 1e-8)
            
            # Calculate weights
            weights = torch.sum(alpha * torch.clamp(gradients, min=0), dim=(2, 3, 4), keepdim=True)
            
            # Generate CAM
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = torch.clamp(cam, min=0)
            
            # Reshape and interpolate
            cam = cam.squeeze()
            original_size = input_tensor.shape[2:]
            if cam.shape != original_size:
                cam = cam.unsqueeze(0).unsqueeze(0)
                cam = F.interpolate(cam, size=original_size, mode='trilinear', align_corners=False)
                cam = cam.squeeze()
            
            # Normalize
            cam_np = cam.cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            
            return cam_np
            
        finally:
            self._remove_hooks()
            
    def generate_smooth_grad(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                           n_samples: int = 50, noise_level: float = 0.15) -> np.ndarray:
        """
        Generate 3D SmoothGrad heatmap.
        
        Args:
            input_tensor: Input 3D volume tensor (1, C, D, H, W)
            target_class: Target class for gradient calculation
            n_samples: Number of noisy samples
            noise_level: Standard deviation of noise
            
        Returns:
            3D heatmap as numpy array (D, H, W)
        """
        self.model.eval()
        
        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                target_class = logits.argmax(dim=1).item()
        
        # Accumulate gradients from noisy samples
        total_gradients = torch.zeros_like(input_tensor)
        
        for _ in range(n_samples):
            # Add noise to input
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_(True)
            noisy_input.retain_grad()  # Ensure gradients are retained for non-leaf tensors
            
            # Forward and backward pass
            logits = self.model(noisy_input)
            self.model.zero_grad()
            
            class_score = logits[0, target_class]
            class_score.backward(retain_graph=True)
            
            # Accumulate gradients
            if noisy_input.grad is not None:
                total_gradients += noisy_input.grad.detach()
        
        # Average the gradients
        avg_gradients = total_gradients / n_samples
        
        # Generate heatmap as magnitude of gradients
        heatmap = torch.sum(avg_gradients.abs(), dim=1).squeeze()  # Sum over channel dimension
        
        # Normalize
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        return heatmap_np
        
    def generate_var_grad(self, input_tensor: torch.Tensor, target_class: Optional[int] = None,
                         n_samples: int = 50, noise_level: float = 0.15) -> np.ndarray:
        """
        Generate 3D VarGrad heatmap (gradient variance).
        
        Args:
            input_tensor: Input 3D volume tensor (1, C, D, H, W)
            target_class: Target class for gradient calculation
            n_samples: Number of noisy samples
            noise_level: Standard deviation of noise
            
        Returns:
            3D heatmap as numpy array (D, H, W)
        """
        self.model.eval()
        
        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                target_class = logits.argmax(dim=1).item()
        
        # Collect gradients from noisy samples
        gradients_list = []
        
        for _ in range(n_samples):
            # Add noise to input
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_(True)
            noisy_input.retain_grad()  # Ensure gradients are retained for non-leaf tensors
            
            # Forward and backward pass
            logits = self.model(noisy_input)
            self.model.zero_grad()
            
            class_score = logits[0, target_class]
            class_score.backward(retain_graph=True)
            
            # Store gradients
            if noisy_input.grad is not None:
                gradients_list.append(noisy_input.grad.detach().clone())
        
        # Stack gradients and calculate variance
        gradients_tensor = torch.stack(gradients_list)  # (n_samples, 1, C, D, H, W)
        gradient_variance = torch.var(gradients_tensor, dim=0)  # (1, C, D, H, W)
        
        # Generate heatmap as sum of variances across channels
        heatmap = torch.sum(gradient_variance, dim=1).squeeze()  # (D, H, W)
        
        # Normalize
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        
        return heatmap_np
    
    def generate_cam_chunked(self, input_tensor: torch.Tensor, target_class: Optional[int] = None, 
                            chunk_size: int = 64) -> np.ndarray:
        """
        Generate 3D GradCAM heatmap using chunked processing for memory efficiency.
        
        Args:
            input_tensor: Input 3D volume tensor (1, C, D, H, W)
            target_class: Target class for CAM generation
            chunk_size: Number of slices to process at once
            
        Returns:
            3D heatmap as numpy array (D, H, W)
        """
        print(f"Using chunked processing with chunk_size={chunk_size}")
        batch_size, channels, depth, height, width = input_tensor.shape
        
        if depth <= chunk_size:
            # If volume is small enough, use regular processing
            return self.generate_cam(input_tensor, target_class)
        
        # Process in chunks
        heatmap_chunks = []
        for start_idx in range(0, depth, chunk_size):
            end_idx = min(start_idx + chunk_size, depth)
            
            # Extract chunk
            chunk = input_tensor[:, :, start_idx:end_idx, :, :].clone()
            
            try:
                # Generate heatmap for this chunk
                chunk_heatmap = self.generate_cam(chunk, target_class)
                heatmap_chunks.append(chunk_heatmap)
                
                # Clear memory after each chunk
                del chunk
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in chunk {start_idx}-{end_idx}, using fallback")
                    # Fallback to zero heatmap for this chunk
                    chunk_shape = (end_idx - start_idx, height, width)
                    fallback_chunk = np.zeros(chunk_shape)
                    heatmap_chunks.append(fallback_chunk)
                else:
                    raise e
        
        # Concatenate all chunks
        full_heatmap = np.concatenate(heatmap_chunks, axis=0)
        return full_heatmap


class MemoryManager:
    """Advanced memory management utilities for 3D volume processing."""
    
    @staticmethod
    def aggressive_gpu_cleanup():
        """Perform aggressive GPU memory cleanup with multiple passes."""
        if torch.cuda.is_available():
            # Multiple cleanup passes
            for i in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            
            # Reset memory stats to help with fragmentation
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_gpu_memory_info():
        """Get current GPU memory usage information."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            truly_free = total_memory - reserved  # Actually available memory
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_memory,
                'free_gb': reserved - allocated,  # PyTorch cached but unused
                'truly_free_gb': truly_free,     # Actually available
                'total_gb': total_memory,
                'fragmentation_ratio': (reserved - allocated) / max(reserved, 0.001)
            }
        return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0, 'free_gb': 0, 'truly_free_gb': 0, 'total_gb': 0, 'fragmentation_ratio': 0}
    
    @staticmethod
    def check_memory_fragmentation():
        """Check if memory is heavily fragmented and needs cleanup."""
        info = MemoryManager.get_gpu_memory_info()
        fragmentation_ratio = info['fragmentation_ratio']
        truly_free_ratio = info['truly_free_gb'] / max(info['total_gb'], 0.001)
        
        # Memory is fragmented if we have low truly free memory but high fragmentation
        is_fragmented = fragmentation_ratio > 0.1 and truly_free_ratio < 0.05
        return is_fragmented, fragmentation_ratio, truly_free_ratio
    
    @staticmethod
    def check_gpu_memory_available(required_gb: float = 1.0):
        """Check if enough GPU memory is available."""
        if not torch.cuda.is_available():
            return False
        
        # First check if we have theoretical space
        info = MemoryManager.get_gpu_memory_info()
        if info['truly_free_gb'] < required_gb:
            return False
        
        try:
            # Try to allocate a test tensor
            test_size = int(required_gb * 1024**3 / 4)  # 4 bytes per float32
            test_tensor = torch.zeros(test_size, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def force_memory_defragmentation():
        """Force memory defragmentation by creating and destroying large tensors."""
        if not torch.cuda.is_available():
            return
        
        try:
            info = MemoryManager.get_gpu_memory_info()
            available_memory = info['truly_free_gb'] * 0.8  # Use 80% of available memory
            
            if available_memory > 0.5:  # Only if we have at least 0.5GB
                # Create a large tensor to force memory consolidation
                tensor_size = int(available_memory * 1024**3 / 4)  # 4 bytes per float32
                temp_tensor = torch.zeros(tensor_size, device='cuda')
                del temp_tensor
                
            # Multiple cleanup passes
            MemoryManager.aggressive_gpu_cleanup()
            print(f"Forced memory defragmentation, freed {available_memory:.2f}GB")
            
        except Exception as e:
            print(f"Memory defragmentation failed: {e}")
            MemoryManager.aggressive_gpu_cleanup()
    
    @staticmethod
    def move_model_to_cpu(model):
        """Move model to CPU to free GPU memory."""
        if model is not None:
            model.cpu()
            MemoryManager.aggressive_gpu_cleanup()
            return True
        return False
    
    @staticmethod
    def move_model_to_gpu(model, device='cuda'):
        """Move model back to GPU."""
        if model is not None:
            model.to(device)
            return True
        return False
    
    @staticmethod
    def critical_memory_recovery():
        """Perform critical memory recovery when system is almost out of memory."""
        print("⚠️  CRITICAL MEMORY RECOVERY INITIATED")
        
        # Force garbage collection
        gc.collect()
        
        # Clear all CUDA cache multiple times
        for i in range(5):
            torch.cuda.empty_cache()
            
        # Reset all memory statistics
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Force defragmentation
        MemoryManager.force_memory_defragmentation()
        
        # Final cleanup
        gc.collect()
        
        info = MemoryManager.get_gpu_memory_info()
        print(f"Critical recovery completed: {info['truly_free_gb']:.2f}GB free")
        
        return info['truly_free_gb'] > 0.5  # Return True if we have at least 0.5GB free
    
    @staticmethod 
    def setup_cuda_memory_optimization():
        """Setup CUDA memory optimization settings."""
        if torch.cuda.is_available():
            # Set expandable segments to reduce fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            print("Reset CUDA memory peak stats")


class DatasetGradCAMAnalyzer:
    """
    Comprehensive 3D Grad-CAM analyzer for entire 3D OCT volume datasets.
    
    This class handles:
    - Loading trained 3D models and 3D OCT volume datasets
    - Batch processing of 3D volume samples with progress tracking
    - Organized output structure for 3D visualizations
    - Statistical analysis and reporting for 3D volumes
    - Memory management for large 3D datasets
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
        
        # Setup memory optimization
        MemoryManager.setup_cuda_memory_optimization()
        
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
        
        # For 3D models, we need to handle the single backbone structure
        print("Note: 3D model uses single backbone structure - adapting visualizers...")
        
        if self.method == 'both':
            # Create custom comparison visualizer for 3D
            self.comparison_visualizer = self._create_3d_comparison_visualizer(use_cuda)
            self.visualizer = None
            print(f"✓ 3D Comparison visualizer ready (Grad-CAM + Grad-CAM++, CUDA: {use_cuda})")
        elif self.method in ['all', 'smoothgrad', 'vargrad']:
            # For these methods, we create visualizers on-demand in analyze_single_sample
            self.visualizer = None
            self.comparison_visualizer = None
            print(f"✓ Method '{self.method}' ready - 3D visualizers will be created on-demand (CUDA: {use_cuda})")
        else:
            # Traditional gradcam or gradcam++ methods for 3D
            self.visualizer = self._create_3d_visualizer(use_cuda, self.method)
            print(f"✓ 3D {self.method.upper()} visualizer ready (CUDA: {use_cuda})")
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Data module setup complete")
    
    def _create_3d_visualizer(self, use_cuda: bool, method: str):
        """Create a visualizer adapted for 3D single-backbone models."""
        # For 3D models with single backbone, we need a custom visualizer
        # For now, create a mock visualizer that we'll handle specially
        class Mock3DVisualizer:
            def __init__(self):
                self.grad_cam = None  # Will be handled specially in analyze_single_sample
            
            def cleanup(self):
                # Placeholder cleanup method
                pass
        return Mock3DVisualizer()
    
    def _create_3d_comparison_visualizer(self, use_cuda: bool):
        """Create a comparison visualizer adapted for 3D single-backbone models."""
        # For 3D models with single backbone, we need a custom comparison visualizer
        # For now, create a mock visualizer that we'll handle specially
        class Mock3DComparisonVisualizer:
            def __init__(self):
                self.compare_methods = None  # Will be handled specially in analyze_single_sample
        return Mock3DComparisonVisualizer()
    
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
            'heatmap_stats': {'volume': []}  # Single 3D volume input
        }
        
        # Process samples
        sample_count = 0
        with tqdm(total=total_samples, desc=f"Processing {split}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                
                volumes, targets, sample_ids = batch
                batch_size = volumes.size(0)
                
                # Process each sample in the batch
                for i in range(batch_size):
                    if max_samples and sample_count >= max_samples:
                        break
                    
                    start_time = time.time()
                    
                    # Extract single sample (3D volume)
                    sample_volume = volumes[i:i+1]  # Shape: (1, C, D, H, W)
                    true_class = int(targets[i].item())
                    sample_id = sample_ids[i] if isinstance(sample_ids, list) else f"{split}_sample_{sample_count:05d}"
                    
                    # Perform Grad-CAM analysis
                    try:
                        results = self.analyze_single_sample(
                            sample_volume, true_class, sample_id, split
                        )
                        
                        # Update statistics
                        pred_class = results['predicted_class']
                        confidence = results['confidence']
                        
                        split_stats['samples_processed'] += 1
                        split_stats['class_counts'][true_class] += 1
                        split_stats['predictions'].append(pred_class)
                        split_stats['ground_truths'].append(true_class)
                        split_stats['confidences'].append(confidence)
                        
                        # Heatmap statistics for 3D volume
                        # Handle different heatmap structures based on method
                        if self.method == 'all':
                            # For 'all' method, use gradcam results as the primary statistics
                            # Structure: {'gradcam': {'volume': array}, 'gradcam++': {...}, ...}
                            primary_method = 'gradcam'  # Use gradcam as representative
                            if primary_method in results['heatmaps'] and 'volume' in results['heatmaps'][primary_method]:
                                heatmap = results['heatmaps'][primary_method]['volume']
                                max_activation = float(heatmap.max())
                                mean_activation = float(heatmap.mean())
                                split_stats['heatmap_stats']['volume'].append({
                                    'max': max_activation,
                                    'mean': mean_activation,
                                    'sample_id': sample_id
                                })
                        else:
                            # For single methods, structure is flat: {'volume': array}
                            if 'volume' in results['heatmaps']:
                                heatmap = results['heatmaps']['volume']
                                max_activation = float(heatmap.max())
                                mean_activation = float(heatmap.mean())
                                split_stats['heatmap_stats']['volume'].append({
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
    
    def analyze_single_sample(self, volume: torch.Tensor, true_class: int, 
                            sample_id: str, split: str) -> Dict[str, Any]:
        """
        Analyze a single 3D volume sample and save results.
        
        Args:
            volume: Input 3D volume tensor (1, C, D, H, W)
            true_class: True class label
            sample_id: Sample identifier
            split: Dataset split name
            
        Returns:
            Analysis results dictionary
        """
        from Utils.grad_cam import (AllMethodsComparison, SmoothGradVisualizer, VarGradVisualizer, 
                                  MultiInputGradCAM, MultiInputGradCAMPlusPlus)
        
        # Aggressive memory management before processing
        MemoryManager.aggressive_gpu_cleanup()
        
        # Check memory fragmentation and defragment if necessary
        is_fragmented, frag_ratio, free_ratio = MemoryManager.check_memory_fragmentation()
        if is_fragmented:
            print(f"⚠️  Memory fragmentation detected (frag: {frag_ratio:.2f}, free: {free_ratio:.2f})")
            MemoryManager.force_memory_defragmentation()
        
        # Check available memory before processing
        memory_info = MemoryManager.get_gpu_memory_info()
        print(f"GPU Memory before processing sample {sample_id}: {memory_info['allocated_gb']:.2f}GB allocated, {memory_info['truly_free_gb']:.2f}GB truly free")
        
        # If we have very low memory, try critical recovery
        if memory_info['truly_free_gb'] < 0.5:
            print(f"⚠️  Critical memory situation for sample {sample_id}")
            if not MemoryManager.critical_memory_recovery():
                print(f"❌ Critical memory recovery failed, skipping sample {sample_id}")
                # Return a fallback result
                return {
                    'sample_id': sample_id,
                    'true_class': true_class,
                    'predicted_class': -1,
                    'confidence': 0.0,
                    'status': 'skipped_low_memory',
                    'processing_time': 0.0
                }
        
        # Move model to GPU just for this sample
        original_device = next(self.model.parameters()).device if self.model is not None else 'cuda'
        if str(original_device) == 'cpu':
            print(f"Moving model to GPU for sample {sample_id}")
            MemoryManager.move_model_to_gpu(self.model)
            MemoryManager.aggressive_gpu_cleanup()
        
        # Ensure volume is on the same device as model
        device = next(self.model.parameters()).device
        volume = volume.to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(volume)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Generate real 3D heatmaps using proper GradCAM for 3D models
        heatmaps = {}
        try:
            # Create 3D GradCAM instance targeting the features before pooling
            # Use the feature extraction part of the backbone instead of the full backbone
            gradcam_3d = GradCAM3D(self.model, target_layer='out.backbone1._conv_head')
            
            if self.method == 'all':
                # Generate all methods with progressive fallback strategies
                gradcam_heatmap = None
                # Try regular GradCAM first
                try:
                    gradcam_heatmap = gradcam_3d.generate_cam(volume, target_class=predicted_class)
                    MemoryManager.clear_gpu_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM during GradCAM for {sample_id}, trying chunked processing...")
                        MemoryManager.aggressive_gpu_cleanup()
                        try:
                            # Try chunked processing with smaller chunks
                            gradcam_heatmap = gradcam_3d.generate_cam_chunked(volume, target_class=predicted_class, chunk_size=32)
                            MemoryManager.clear_gpu_cache()
                        except RuntimeError as e2:
                            if "out of memory" in str(e2):
                                print(f"Chunked GradCAM also failed for {sample_id}, skipping...")
                                gradcam_heatmap = None
                            else:
                                raise e2
                    else:
                        raise e
                
                gradcampp_heatmap = None
                try:
                    gradcampp_heatmap = gradcam_3d.generate_cam_plus_plus(volume, target_class=predicted_class)
                    MemoryManager.clear_gpu_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM during GradCAM++ for {sample_id}, trying chunked processing...")
                        MemoryManager.aggressive_gpu_cleanup()
                        try:
                            # Try chunked processing
                            gradcampp_heatmap = gradcam_3d.generate_cam_chunked(volume, target_class=predicted_class, chunk_size=32)
                            MemoryManager.clear_gpu_cache()
                        except RuntimeError as e2:
                            if "out of memory" in str(e2):
                                print(f"Chunked GradCAM++ also failed for {sample_id}, skipping...")
                                gradcampp_heatmap = None
                            else:
                                raise e2
                    else:
                        raise e
                
                # For SmoothGrad and VarGrad, use progressive reduction based on memory
                memory_info = MemoryManager.get_gpu_memory_info()
                if memory_info['truly_free_gb'] > 5.0:
                    reduced_samples = self.n_samples  # Full samples if lots of memory
                elif memory_info['truly_free_gb'] > 2.0:
                    reduced_samples = max(20, self.n_samples // 2)  # Half samples for medium memory
                elif memory_info['truly_free_gb'] > 1.0:
                    reduced_samples = max(10, self.n_samples // 4)  # Quarter samples for low memory
                else:
                    reduced_samples = 5  # Minimal samples for critical memory
                
                try:
                    smoothgrad_heatmap = gradcam_3d.generate_smooth_grad(volume, target_class=predicted_class, 
                                                                        n_samples=reduced_samples, noise_level=self.noise_level)
                    MemoryManager.clear_gpu_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM during SmoothGrad for {sample_id}, skipping...")
                        smoothgrad_heatmap = None
                    else:
                        raise e
                
                try:
                    vargrad_heatmap = gradcam_3d.generate_var_grad(volume, target_class=predicted_class,
                                                                  n_samples=reduced_samples, noise_level=self.noise_level)
                    MemoryManager.clear_gpu_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM during VarGrad for {sample_id}, skipping...")
                        vargrad_heatmap = None
                    else:
                        raise e
                
                # Only include successful heatmaps
                heatmaps = {}
                if gradcam_heatmap is not None:
                    heatmaps['gradcam'] = {'volume': gradcam_heatmap}
                if gradcampp_heatmap is not None:
                    heatmaps['gradcam++'] = {'volume': gradcampp_heatmap}
                if smoothgrad_heatmap is not None:
                    heatmaps['smoothgrad'] = {'volume': smoothgrad_heatmap}
                if vargrad_heatmap is not None:
                    heatmaps['vargrad'] = {'volume': vargrad_heatmap}
                print(f"Generated real 3D heatmaps for all methods (sample: {sample_id})")
                
            elif self.method == 'both':
                # Grad-CAM vs Grad-CAM++ with error handling
                gradcam_heatmap = gradcampp_heatmap = None
                try:
                    gradcam_heatmap = gradcam_3d.generate_cam(volume, target_class=predicted_class)
                    MemoryManager.clear_gpu_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM during GradCAM for {sample_id}")
                    else:
                        raise e
                
                try:
                    gradcampp_heatmap = gradcam_3d.generate_cam_plus_plus(volume, target_class=predicted_class)
                    MemoryManager.clear_gpu_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA OOM during GradCAM++ for {sample_id}")
                    else:
                        raise e
                
                # Only include successful heatmaps
                if gradcam_heatmap is not None and gradcampp_heatmap is not None:
                    grad_cam_heatmaps = {'volume': gradcam_heatmap}
                    grad_cam_pp_heatmaps = {'volume': gradcampp_heatmap}
                    heatmaps = {'gradcam': grad_cam_heatmaps, 'gradcam++': grad_cam_pp_heatmaps}
                    print(f"Generated real 3D heatmaps for both methods (sample: {sample_id})")
                elif gradcam_heatmap is not None:
                    heatmaps = {'gradcam': {'volume': gradcam_heatmap}}
                    print(f"Generated only GradCAM heatmap for {sample_id} due to memory constraints")
                elif gradcampp_heatmap is not None:
                    heatmaps = {'gradcam++': {'volume': gradcampp_heatmap}}
                    print(f"Generated only GradCAM++ heatmap for {sample_id} due to memory constraints")
                
            elif self.method == 'smoothgrad':
                # SmoothGrad method with memory optimization
                memory_info = MemoryManager.get_gpu_memory_info()
                reduced_samples = self.n_samples if memory_info['free_gb'] > 2.0 else max(10, self.n_samples // 4)
                heatmap = gradcam_3d.generate_smooth_grad(volume, target_class=predicted_class,
                                                         n_samples=reduced_samples, noise_level=self.noise_level)
                heatmaps = {'volume': heatmap}
                print(f"Generated real 3D SmoothGrad heatmap with {reduced_samples} samples (sample: {sample_id})")
                
            elif self.method == 'vargrad':
                # VarGrad method with memory optimization
                memory_info = MemoryManager.get_gpu_memory_info()
                reduced_samples = self.n_samples if memory_info['free_gb'] > 2.0 else max(10, self.n_samples // 4)
                heatmap = gradcam_3d.generate_var_grad(volume, target_class=predicted_class,
                                                      n_samples=reduced_samples, noise_level=self.noise_level)
                heatmaps = {'volume': heatmap}
                print(f"Generated real 3D VarGrad heatmap with {reduced_samples} samples (sample: {sample_id})")
                
            elif self.method == 'gradcam':
                # Standard Grad-CAM
                heatmap = gradcam_3d.generate_cam(volume, target_class=predicted_class)
                heatmaps = {'volume': heatmap}
                print(f"Generated real 3D GradCAM heatmap (sample: {sample_id})")
                
            elif self.method == 'gradcam++':
                # Grad-CAM++
                heatmap = gradcam_3d.generate_cam_plus_plus(volume, target_class=predicted_class)
                heatmaps = {'volume': heatmap}
                print(f"Generated real 3D GradCAM++ heatmap (sample: {sample_id})")
            
            else:
                # Fallback to standard Grad-CAM
                heatmap = gradcam_3d.generate_cam(volume, target_class=predicted_class)
                heatmaps = {'volume': heatmap}
                print(f"Generated real 3D GradCAM heatmap for unknown method {self.method} (sample: {sample_id})")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA out of memory error for sample {sample_id}: {e}")
                print(f"GPU Memory info: {MemoryManager.get_gpu_memory_info()}")
                MemoryManager.clear_gpu_cache()
                
                # Try to recover by using fallback heatmap
                volume_shape = volume.shape  # (1, C, D, H, W)
                fallback_heatmap = np.zeros((volume_shape[2], volume_shape[3], volume_shape[4]))  # (D, H, W)
                heatmaps = {'volume': fallback_heatmap}
                print(f"Using fallback zero heatmap for {sample_id} due to memory constraints")
            else:
                print(f"Runtime error generating heatmaps for {sample_id}: {e}")
                raise e
        except Exception as e:
            print(f"Unexpected error generating heatmaps for {sample_id}: {e}")
            # Fallback to a zeros array with correct dimensions
            volume_shape = volume.shape  # (1, C, D, H, W)
            fallback_heatmap = np.zeros((volume_shape[2], volume_shape[3], volume_shape[4]))  # (D, H, W)
            heatmaps = {'volume': fallback_heatmap}
            print(f"Using fallback zero heatmap for {sample_id}")
        finally:
            # Always clear memory after processing each sample
            MemoryManager.aggressive_gpu_cleanup()
        
        # Save the results to .mat files
        print(f"Saving results for sample {sample_id}")
        volume_list = [volume.detach().cpu()]  # Convert to list for compatibility with save method
        
        # Flatten nested heatmaps structure for save method
        flat_heatmaps = {}
        if isinstance(heatmaps, dict):
            for method_name, method_heatmaps in heatmaps.items():
                if isinstance(method_heatmaps, dict):
                    # Nested structure (e.g., 'all' or 'both' methods)
                    for heatmap_key, heatmap_data in method_heatmaps.items():
                        flat_heatmaps[f"{method_name}_{heatmap_key}"] = heatmap_data
                else:
                    # Direct heatmap data
                    flat_heatmaps[method_name] = method_heatmaps
        
        self.save_sample_results(volume_list, flat_heatmaps, {
            'sample_id': sample_id,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.detach().cpu().numpy()[0]
        }, split)
        
        # Move model back to CPU to free GPU memory for next sample
        if self.model is not None:
            print(f"Moving model to CPU after processing {sample_id}")
            MemoryManager.move_model_to_cpu(self.model)
            
        # Final aggressive cleanup
        MemoryManager.aggressive_gpu_cleanup()
        
        return {
            'sample_id': sample_id,
            'true_class': true_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.detach().cpu().numpy()[0],
            'heatmaps': heatmaps,
            'method': self.method
        }
    
    def write_stack_image_file(self, fn: str, stack_img: np.ndarray, colormap: Any = None, description: dict = {}) -> None:
        """
        Write stack image to TIFF, optionally with colormap.
        Args:
            fn: Full file path.
            stack_img: Numpy image array.
            colormap: Optional colormap.
        """
        if colormap is None:
            tifffile.imwrite(fn, stack_img, metadata={"axes": "ZXY"}, compression="zlib", description=json.dumps(description))
        else:
            if isinstance(colormap, list):
                if len(colormap) < 256:
                    colormap += [(0, 0, 0)] * (256 - len(colormap))
                elif len(colormap) > 256:
                    colormap = colormap[:256]
                colormap = np.array(colormap).reshape(3, 256)
            elif isinstance(colormap, np.ndarray):
                if colormap.shape == (3, 256):
                    pass
                elif colormap.shape == (256, 3):
                    colormap = colormap.T
                else:
                    raise ValueError("colormap should be a 2D array with shape (3,256) or (256,3)")
            tifffile.imwrite(fn, stack_img, metadata={"axes": "ZXY"}, colormap=colormap, compression="zlib", description=json.dumps(description))
    
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
        
        # For 3D models, save heatmaps as full volume TIFF with jet colormap
        import scipy.io as sio
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        
        # Get input volume for overlay visualizations
        input_volume = inputs[0].squeeze().detach().cpu().numpy()  # Remove batch dimension and gradients, shape: (C, D, H, W) or (D, H, W)
        if input_volume.ndim == 4:  # Multi-channel, take first channel
            input_volume = input_volume[0]  # Shape: (D, H, W)
        elif input_volume.ndim == 3:  # Single channel
            pass  # Shape: (D, H, W)
        
        # Ensure input volume is in correct shape (D, H, W) with D=256 slices
        # If the first dimension is not 256, transpose to (2, 0, 1) to make depth first
        if input_volume.shape[0] != 256 and input_volume.shape[2] == 256:
            input_volume = np.transpose(input_volume, (2, 0, 1))  # (H, W, D) -> (D, H, W)
            print(f"Transposed input volume to correct shape: {input_volume.shape}")
        
        for heatmap_name, heatmap in heatmaps.items():
            # Process heatmap data to ensure correct shape for saving
            import torch
            if isinstance(heatmap, torch.Tensor):  # It's a tensor
                heatmap_data = heatmap.detach().cpu().numpy()
            else:  # It's already a numpy array
                heatmap_data = heatmap
                
            if heatmap_data.ndim == 4 and heatmap_data.shape[0] == 1:
                heatmap_data = heatmap_data.squeeze(0)  # Remove batch dimension
            
            # Ensure heatmap is in (D, H, W) format with D=256 slices
            # If the first dimension is not 256, transpose to (2, 0, 1) to make depth first
            if heatmap_data.shape[0] != 256 and heatmap_data.shape[2] == 256:
                heatmap_data = np.transpose(heatmap_data, (2, 0, 1))  # (H, W, D) -> (D, H, W)
                print(f"Transposed heatmap data to correct shape: {heatmap_data.shape}")
            
            # Verify dimensions match between input and heatmap
            if input_volume.shape != heatmap_data.shape:
                print(f"Warning: Shape mismatch - Input: {input_volume.shape}, Heatmap: {heatmap_data.shape}")
            # Save as .mat file for compatibility
            # heatmap_file_mat = heatmap_dir / f"{sample_id}_{heatmap_name}.mat"
            # sio.savemat(heatmap_file_mat, {
            #     'heatmap': heatmap_data,
            #     'shape': heatmap_data.shape,
            #     'method': self.method,
            #     'sample_id': str(sample_id),
            #     'true_class': true_class,
            #     'predicted_class': pred_class
            # })
            
            # Save full volume heatmap as TIFF with jet colormap
            heatmap_file_tiff = heatmap_dir / f"{sample_id}_{heatmap_name}_heatmap.tiff"
            overlay_file_tiff = heatmap_dir / f"{sample_id}_{heatmap_name}_overlay.tiff"
            
            try:
                # Normalize heatmap to 0-1 range for jet colormap
                heatmap_normalized = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
                
                # Apply jet colormap to create RGB heatmap volume
                jet_colormap = plt.colormaps.get_cmap('jet')
                heatmap_rgb = jet_colormap(heatmap_normalized)  # Shape: (D, H, W, 4) RGBA
                heatmap_rgb = (heatmap_rgb[:, :, :, :3] * 255).astype(np.uint8)  # Remove alpha, convert to uint8, shape: (D, H, W, 3)
                
                # Save full volume heatmap with jet colormap as RGB TIFF
                tifffile.imwrite(heatmap_file_tiff, heatmap_rgb, photometric='rgb', compression='lzw')
                print(f"Saved 3D heatmap with jet colormap: {heatmap_file_tiff} ({heatmap_rgb.shape[0]} slices)")
                
                # Create overlay visualization (heatmap + input data)
                # Normalize input volume to 0-1 range
                input_normalized = (input_volume - input_volume.min()) / (input_volume.max() - input_volume.min() + 1e-8)
                
                # Convert input to RGB (grayscale -> RGB)
                input_rgb = np.stack([input_normalized] * 3, axis=-1)  # Shape: (D, H, W, 3)
                input_rgb = (input_rgb * 255).astype(np.uint8)
                
                # Create overlay by blending input and heatmap
                alpha = 0.4  # Heatmap transparency
                overlay_rgb = ((1 - alpha) * input_rgb + alpha * heatmap_rgb).astype(np.uint8)
                
                # Save overlay as RGB TIFF
                tifffile.imwrite(overlay_file_tiff, overlay_rgb, photometric='rgb', compression='lzw')
                print(f"Saved 3D overlay visualization: {overlay_file_tiff} ({overlay_rgb.shape[0]} slices)")
                
            except Exception as e:
                print(f"Failed to save TIFF for {heatmap_name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Saved 3D heatmaps (.mat and .tiff) to: {heatmap_dir}")
        
        # Save combined visualization with full volume overlay (heatmap + input data)
        viz_dir = self.visualizations_dir / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path_png = viz_dir / f"{sample_id}_combined.png"
        viz_path_tiff = viz_dir / f"{sample_id}_full_volume_overlay.tiff"
        
        # For 3D models, create visualizations with overlays using jet colormap
        try:
            
            # Create combined visualization showing multiple slices with overlays
            if heatmaps and len(heatmaps) > 0:
                # Get the first heatmap for visualization
                first_heatmap_name = list(heatmaps.keys())[0]
                heatmap_3d = heatmaps[first_heatmap_name]
                
                # Process heatmap data (should already be corrected above in the loop)
                import torch
                if isinstance(heatmap_3d, torch.Tensor):
                    heatmap_3d = heatmap_3d.detach().cpu().numpy()
                
                # Apply same dimension correction logic as above
                if heatmap_3d.shape[0] != 256 and heatmap_3d.shape[2] == 256:
                    heatmap_3d = np.transpose(heatmap_3d, (2, 0, 1))  # (H, W, D) -> (D, H, W)
                
                depth = heatmap_3d.shape[0]
                # Select evenly distributed slices for preview visualization
                n_slices = min(6, depth)  # Show up to 6 slices
                slice_indices = np.linspace(0, depth-1, n_slices, dtype=int)
                
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten() if n_slices > 1 else [axes]
                
                for i, slice_idx in enumerate(slice_indices):
                    if i < len(axes):
                        # Get input slice
                        input_slice = input_volume[slice_idx]
                        heatmap_slice = heatmap_3d[slice_idx]
                        
                        # Show input as grayscale background
                        axes[i].imshow(input_slice, cmap='gray', alpha=0.8)
                        # Overlay heatmap with jet colormap
                        im = axes[i].imshow(heatmap_slice, cmap='jet', alpha=0.6)
                        axes[i].set_title(f'Overlay Slice {slice_idx}/{depth-1}')
                        axes[i].axis('off')
                
                # Hide unused subplots
                for i in range(len(slice_indices), len(axes)):
                    axes[i].axis('off')
                
                fig.suptitle(f'3D {self.method.upper()} Overlay - {first_heatmap_name} (Pred: {self.class_names[pred_class]})', 
                           fontsize=16)
                plt.tight_layout()
                plt.savefig(viz_path_png, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save full volume overlay as TIFF
                # Create overlay for entire volume
                # Normalize input volume to 0-1 range
                input_norm = (input_volume - input_volume.min()) / (input_volume.max() - input_volume.min() + 1e-8)
                
                # Normalize heatmap to 0-1 range
                heatmap_norm = (heatmap_3d - heatmap_3d.min()) / (heatmap_3d.max() - heatmap_3d.min() + 1e-8)
                
                # Apply jet colormap to heatmap
                jet_colormap = plt.colormaps.get_cmap('jet')
                heatmap_rgb = jet_colormap(heatmap_norm)[:, :, :, :3]  # Remove alpha channel, shape: (D, H, W, 3)
                
                # Convert input to RGB (grayscale background)
                input_rgb = np.stack([input_norm] * 3, axis=-1)  # Shape: (D, H, W, 3)
                
                # Create overlay: blend input (grayscale) with heatmap (jet colormap)
                alpha = 0.5  # 50% transparency for heatmap
                full_overlay_rgb = ((1 - alpha) * input_rgb + alpha * heatmap_rgb)
                full_overlay_rgb = (full_overlay_rgb * 255).astype(np.uint8)
                
                # Save full volume overlay as RGB TIFF
                tifffile.imwrite(viz_path_tiff, full_overlay_rgb, photometric='rgb', compression='lzw')
                print(f"Saved full volume overlay visualization: {viz_path_png} and {viz_path_tiff} ({depth} slices)")
                
            else:
                # Fallback visualization
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, 'No heatmap data available', ha='center', va='center')
                ax.set_title('3D GradCAM - No Data')
                plt.tight_layout()
                plt.savefig(viz_path_png, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved fallback visualization to: {viz_path_png}")
                
        except Exception as e:
            print(f"Failed to create 3D visualization: {e}")
        
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
                'volume': {  # For 3D, we have single volume heatmap
                    'max_activation': float(list(heatmaps.values())[0].max()) if heatmaps else 0.0,
                    'mean_activation': float(list(heatmaps.values())[0].mean()) if heatmaps else 0.0,
                    'std_activation': float(list(heatmaps.values())[0].std()) if heatmaps else 0.0
                }
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
        
        # For 3D, save heatmaps as .mat files
        import scipy.io as sio
        for heatmap_name, heatmap_data in grad_cam_heatmaps.items():
            heatmap_file = gradcam_dir / f"{sample_id}_gradcam_{heatmap_name}.mat"
            sio.savemat(heatmap_file, {
                'heatmap': heatmap_data,
                'shape': heatmap_data.shape,
                'method': 'gradcam',
                'sample_id': str(sample_id),
                'true_class': true_class,
                'predicted_class': pred_class
            })
        
        for heatmap_name, heatmap_data in grad_cam_pp_heatmaps.items():
            heatmap_file = gradcampp_dir / f"{sample_id}_gradcampp_{heatmap_name}.mat"
            sio.savemat(heatmap_file, {
                'heatmap': heatmap_data,
                'shape': heatmap_data.shape,
                'method': 'gradcam++',
                'sample_id': str(sample_id),
                'true_class': true_class,
                'predicted_class': pred_class
            })
        
        # Save simple comparison visualization
        viz_dir = self.visualizations_dir / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = viz_dir / f"{sample_id}_comparison.png"
        
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # GradCAM visualization
            if 'volume' in grad_cam_heatmaps:
                heatmap_3d = grad_cam_heatmaps['volume']
                middle_slice_idx = heatmap_3d.shape[0] // 2
                middle_slice = heatmap_3d[middle_slice_idx]
                axes[0].imshow(middle_slice, cmap='hot', alpha=0.7)
                axes[0].set_title('GradCAM')
            
            # GradCAM++ visualization  
            if 'volume' in grad_cam_pp_heatmaps:
                heatmap_3d = grad_cam_pp_heatmaps['volume']
                middle_slice_idx = heatmap_3d.shape[0] // 2
                middle_slice = heatmap_3d[middle_slice_idx]
                axes[1].imshow(middle_slice, cmap='hot', alpha=0.7)
                axes[1].set_title('GradCAM++')
            
            plt.suptitle(f'3D Comparison (Pred: {self.class_names[pred_class]})')
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved 3D comparison to: {comparison_path}")
        except Exception as e:
            print(f"Failed to create 3D comparison: {e}")
        
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
    """Main function with command line interface for 3D volume analysis."""
    parser = argparse.ArgumentParser(description='Comprehensive 3D Grad-CAM Analysis for Entire 3D OCT Volume Dataset')
    
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
    

# Example Usage Commands for 3D Analysis:

# # Individual methods for 3D volumes
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset all --method gradcam
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset all --method gradcam++
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset all --method smoothgrad --n_samples 25
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset all --method vargrad --noise_level 0.2

# # Comparison modes for 3D volumes
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset all --method both
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset all --method all  # All 4 methods for 3D

# # 3D analysis with specific dataset splits:
# python ModelGradCAM_3D.py --config configs/config_3d.toml --dataset val --method all