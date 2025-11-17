"""
Grad-CAM and Grad-CAM++ (Gradient-weighted Class Activation Mapping) implementation
for multi-input EfficientNet-based classification models.

This module provides visualization of important regions in each input image
that contribute to the model's classification decision.

Includes both:
- Grad-CAM: Original gradient-weighted activation mapping
- Grad-CAM++: Improved version with pixel-wise weighting for better localization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
from PIL import Image

# OpenCV is optional - fallback to other methods if not available
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("OpenCV not available. Using PIL for image operations.")


class MultiInputGradCAM:
    """
    Grad-CAM implementation for multi-input models.
    
    This class generates Class Activation Maps for models that accept multiple
    input branches, such as the 4-branch EfficientNet classifier.
    
    Args:
        model: The neural network model
        target_layers: List of target layer names for each input branch
        use_cuda: Whether to use CUDA if available
        
    Example:
        >>> grad_cam = MultiInputGradCAM(model, ['backbone1', 'backbone2', 'backbone3', 'backbone4'])
        >>> heatmaps = grad_cam.generate_cam([mnv, fluid, ga, drusen], target_class=1)
    """
    
    def __init__(self, model, target_layers: List[str], use_cuda: bool = True):
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for gradients and activations
        self.gradients = {}
        self.activations = {}
        self.handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(model, input, output):
                self.gradients[name] = output[0].detach()
            return hook
        
        # Register hooks for each target layer
        for layer_name in self.target_layers:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                # Forward hook for activations
                handle_f = layer.register_forward_hook(get_activation(layer_name))
                # Backward hook for gradients
                handle_b = layer.register_full_backward_hook(get_gradient(layer_name))
                self.handles.extend([handle_f, handle_b])
    
    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name from the model."""
        # For NetModule with Classifier inside
        if hasattr(self.model, 'out'):
            classifier = self.model.out
            if hasattr(classifier, layer_name):
                return getattr(classifier, layer_name)
        
        # Direct access
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)
        
        print(f"Warning: Layer '{layer_name}' not found in model")
        return None
    
    def generate_cam(self, inputs: List[torch.Tensor], target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Class Activation Maps for multiple inputs.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Dictionary mapping input names to their heatmaps
        """
        input_names = ['mnv', 'fluid', 'ga', 'drusen']
        
        # Ensure inputs are on the correct device
        inputs = [inp.to(self.device) for inp in inputs]
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        with torch.enable_grad():
            for inp in inputs:
                inp.requires_grad_(True)
            
            outputs = self.model(*inputs)
            
            # Use predicted class if target_class is None
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            class_score = outputs[:, target_class]
            class_score.backward(retain_graph=True)
        
        # Generate CAMs for each input
        heatmaps = {}
        for i, (inp_name, layer_name) in enumerate(zip(input_names, self.target_layers)):
            if layer_name in self.gradients and layer_name in self.activations:
                heatmap = self._compute_cam(
                    self.activations[layer_name],
                    self.gradients[layer_name],
                    inputs[i]
                )
                heatmaps[inp_name] = heatmap
        
        return heatmaps
    
    def _compute_cam(self, activations: torch.Tensor, gradients: torch.Tensor, 
                     original_input: torch.Tensor) -> np.ndarray:
        """
        Compute Class Activation Map from activations and gradients.
        
        Args:
            activations: Feature activations [B, C, H, W]
            gradients: Gradients w.r.t. activations [B, C, H, W]
            original_input: Original input tensor for resizing
            
        Returns:
            Heatmap as numpy array
        """
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()  # [H, W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to original input size
        input_size = original_input.shape[-2:]  # [H, W]
        if cam.shape != input_size:
            if HAS_OPENCV:
                import cv2  # Local import to avoid type checker issues
                cam = cv2.resize(cam, (input_size[1], input_size[0]))
            else:
                # Use PIL for resizing - handle edge cases
                if cam.ndim == 0:  # Scalar case
                    cam = np.ones(input_size) * cam
                elif cam.ndim == 1:  # 1D case
                    cam = np.tile(cam.reshape(-1, 1), (1, input_size[1]))
                    if cam.shape[0] != input_size[0]:
                        cam = np.tile(cam, (input_size[0], 1))
                else:  # 2D case
                    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
                    cam_pil = cam_pil.resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
                    cam = np.array(cam_pil).astype(np.float32) / 255.0
        
        return cam
    
    def visualize_cam(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray],
                      save_path: Optional[Path] = None, target_class: Optional[int] = None) -> None:
        """
        Visualize the original images with their corresponding heatmaps.
        
        Args:
            inputs: List of input tensors
            heatmaps: Dictionary of heatmaps from generate_cam()
            save_path: Path to save the visualization
            target_class: Target class for title
        """
        input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
            # Original image
            img = inputs[i].squeeze().detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] <= 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f'{display_name} Input')
            axes[0, i].axis('off')
            
            # Heatmap overlay
            if inp_name in heatmaps:
                heatmap = heatmaps[inp_name]
                
                # Create overlay
                if img.ndim == 2:  # Grayscale
                    img_colored = plt.cm.get_cmap('gray')(img)[:, :, :3]
                else:
                    img_colored = img
                
                heatmap_colored = plt.cm.get_cmap('jet')(heatmap)[:, :, :3]
                overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{display_name} Grad-CAM')
            else:
                axes[1, i].text(0.5, 0.5, 'No heatmap\navailable', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].set_title(f'{display_name} Grad-CAM')
            
            axes[1, i].axis('off')
        
        title = f'Grad-CAM Visualization'
        if target_class is not None:
            title += f' - Target Class: {target_class}'
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {save_path}")
        
        plt.show()
    
    def save_heatmaps(self, heatmaps: Dict[str, np.ndarray], save_dir: Path,
                      filename_prefix: str = "heatmap") -> None:
        """
        Save individual heatmaps as images.
        
        Args:
            heatmaps: Dictionary of heatmaps
            save_dir: Directory to save heatmaps
            filename_prefix: Prefix for filenames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for inp_name, heatmap in heatmaps.items():
            # Convert to colored heatmap
            colored_heatmap = plt.cm.get_cmap('jet')(heatmap)
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Save
            filename = f"{filename_prefix}_{inp_name}.png"
            filepath = save_dir / filename
            
            if HAS_OPENCV:
                import cv2  # Local import to avoid type checker issues
                cv2.imwrite(str(filepath), cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
            else:
                # Use PIL to save
                heatmap_pil = Image.fromarray(colored_heatmap)
                heatmap_pil.save(str(filepath))
            
        print(f"Individual heatmaps saved to: {save_dir}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.remove_hooks()


class MultiInputGradCAMPlusPlus:
    """
    Grad-CAM++ implementation for multi-input models.
    
    Grad-CAM++ improves upon Grad-CAM by using pixel-wise weighting of gradients
    instead of global average pooling, providing better localization especially
    for multiple objects or complex scenes.
    
    Reference: "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks"
    
    Args:
        model: The neural network model
        target_layers: List of target layer names for each input branch
        use_cuda: Whether to use CUDA if available
        
    Example:
        >>> grad_cam_pp = MultiInputGradCAMPlusPlus(model, ['backbone1', 'backbone2', 'backbone3', 'backbone4'])
        >>> heatmaps = grad_cam_pp.generate_cam([mnv, fluid, ga, drusen], target_class=1)
    """
    
    def __init__(self, model, target_layers: List[str], use_cuda: bool = True):
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for gradients and activations
        self.gradients = {}
        self.activations = {}
        self.handles = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(model, input, output):
                self.gradients[name] = output[0].detach()
            return hook
        
        # Register hooks for each target layer
        for layer_name in self.target_layers:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                # Forward hook for activations
                handle_f = layer.register_forward_hook(get_activation(layer_name))
                # Backward hook for gradients
                handle_b = layer.register_full_backward_hook(get_gradient(layer_name))
                self.handles.extend([handle_f, handle_b])
    
    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name from the model."""
        # For NetModule with Classifier inside
        if hasattr(self.model, 'out'):
            classifier = self.model.out
            if hasattr(classifier, layer_name):
                return getattr(classifier, layer_name)
        
        # Direct access
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)
        
        print(f"Warning: Layer '{layer_name}' not found in model")
        return None
    
    def generate_cam(self, inputs: List[torch.Tensor], target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM++ Class Activation Maps for multiple inputs.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Dictionary mapping input names to their heatmaps
        """
        input_names = ['mnv', 'fluid', 'ga', 'drusen']
        
        # Ensure inputs are on the correct device
        inputs = [inp.to(self.device) for inp in inputs]
        
        # Clear previous gradients and activations
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        with torch.enable_grad():
            for inp in inputs:
                inp.requires_grad_(True)
            
            outputs = self.model(*inputs)
            
            # Use predicted class if target_class is None
            if target_class is None:
                target_class = outputs.argmax(dim=1).item()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            class_score = outputs[:, target_class]
            class_score.backward(retain_graph=True)
        
        # Generate CAMs for each input
        heatmaps = {}
        for i, (inp_name, layer_name) in enumerate(zip(input_names, self.target_layers)):
            if layer_name in self.gradients and layer_name in self.activations:
                heatmap = self._compute_gradcam_plus_plus(
                    self.activations[layer_name],
                    self.gradients[layer_name],
                    inputs[i]
                )
                heatmaps[inp_name] = heatmap
        
        return heatmaps
    
    def _compute_gradcam_plus_plus(self, activations: torch.Tensor, gradients: torch.Tensor, 
                                   original_input: torch.Tensor) -> np.ndarray:
        """
        Compute Grad-CAM++ Class Activation Map from activations and gradients.
        
        Args:
            activations: Feature activations [B, C, H, W]
            gradients: Gradients w.r.t. activations [B, C, H, W]
            original_input: Original input tensor for resizing
            
        Returns:
            Heatmap as numpy array
        """
        # Grad-CAM++ computation
        # alpha_c_k = gradients^2 / (2 * gradients^2 + sum_over_spatial(activations * gradients^3))
        gradients_2 = gradients.pow(2)
        gradients_3 = gradients.pow(3)
        
        # Sum over spatial dimensions for denominator
        alpha_denom = 2.0 * gradients_2 + (activations * gradients_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.clamp(alpha_denom, min=1e-7)  # Avoid division by zero
        
        # Compute alpha weights
        alpha = gradients_2 / alpha_denom  # [B, C, H, W]
        
        # Compute weights: w_c_k = sum_over_spatial(alpha_c_k * ReLU(gradients))
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()  # [H, W]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to original input size
        input_size = original_input.shape[-2:]  # [H, W]
        if cam.shape != input_size:
            if HAS_OPENCV:
                import cv2  # Local import to avoid type checker issues
                cam = cv2.resize(cam, (input_size[1], input_size[0]))
            else:
                # Use PIL for resizing - handle edge cases
                if cam.ndim == 0:  # Scalar case
                    cam = np.ones(input_size) * cam
                elif cam.ndim == 1:  # 1D case
                    cam = np.tile(cam.reshape(-1, 1), (1, input_size[1]))  
                    if cam.shape[0] != input_size[0]:
                        cam = np.tile(cam, (input_size[0], 1))
                else:  # 2D case
                    cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
                    cam_pil = cam_pil.resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
                    cam = np.array(cam_pil).astype(np.float32) / 255.0
        
        return cam
    
    def visualize_cam(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray],
                      save_path: Optional[Path] = None, target_class: Optional[int] = None) -> None:
        """
        Visualize the original images with their corresponding Grad-CAM++ heatmaps.
        
        Args:
            inputs: List of input tensors
            heatmaps: Dictionary of heatmaps from generate_cam()
            save_path: Path to save the visualization
            target_class: Target class for title
        """
        input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
            # Original image
            img = inputs[i].squeeze().detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] <= 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f'{display_name} Input')
            axes[0, i].axis('off')
            
            # Heatmap overlay
            if inp_name in heatmaps:
                heatmap = heatmaps[inp_name]
                
                # Create overlay
                if img.ndim == 2:  # Grayscale
                    img_colored = plt.cm.get_cmap('gray')(img)[:, :, :3]
                else:
                    img_colored = img
                
                heatmap_colored = plt.cm.get_cmap('jet')(heatmap)[:, :, :3]
                overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{display_name} Grad-CAM++')
            else:
                axes[1, i].text(0.5, 0.5, 'No heatmap\navailable', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].set_title(f'{display_name} Grad-CAM++')
            
            axes[1, i].axis('off')
        
        title = f'Grad-CAM++ Visualization'
        if target_class is not None:
            title += f' - Target Class: {target_class}'
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM++ visualization saved to: {save_path}")
        
        plt.show()
    
    def save_heatmaps(self, heatmaps: Dict[str, np.ndarray], save_dir: Path,
                      filename_prefix: str = "heatmap") -> None:
        """
        Save individual Grad-CAM++ heatmaps as images.
        
        Args:
            heatmaps: Dictionary of heatmaps
            save_dir: Directory to save heatmaps
            filename_prefix: Prefix for filenames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for inp_name, heatmap in heatmaps.items():
            # Convert to colored heatmap
            colored_heatmap = plt.cm.get_cmap('jet')(heatmap)
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Save
            filename = f"{filename_prefix}_{inp_name}.png"
            filepath = save_dir / filename
            
            if HAS_OPENCV:
                import cv2  # Local import to avoid type checker issues
                cv2.imwrite(str(filepath), cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
            else:
                # Use PIL to save
                heatmap_pil = Image.fromarray(colored_heatmap)
                heatmap_pil.save(str(filepath))
            
        print(f"Individual Grad-CAM++ heatmaps saved to: {save_dir}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.remove_hooks()


class GradCAMVisualizer:
    """
    High-level interface for Grad-CAM and Grad-CAM++ visualization and batch processing.
    
    Args:
        model: The neural network model
        use_cuda: Whether to use CUDA if available
        method: Either 'gradcam' or 'gradcam++' to choose the visualization method
    """
    
    def __init__(self, model, use_cuda: bool = True, method: str = 'gradcam'):
        self.model = model
        self.target_layers = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
        self.method = method.lower()
        
        if self.method == 'gradcam':
            self.grad_cam = MultiInputGradCAM(model, self.target_layers, use_cuda)
        elif self.method == 'gradcam++':
            self.grad_cam = MultiInputGradCAMPlusPlus(model, self.target_layers, use_cuda)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose 'gradcam' or 'gradcam++'.")
    
    def analyze_sample(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                      save_dir: Optional[Path] = None, sample_id: str = "sample") -> Dict[str, np.ndarray]:
        """
        Analyze a single sample and generate visualizations.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class for analysis
            save_dir: Directory to save results
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary of heatmaps
        """
        # Generate heatmaps
        heatmaps = self.grad_cam.generate_cam(inputs, target_class)
        
        # Save results if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            method_name = self.method.replace('+', 'plus')  # Handle + in filenames
            viz_path = save_dir / f"{sample_id}_{method_name}_visualization.png"
            self.grad_cam.visualize_cam(inputs, heatmaps, viz_path, target_class)
            
            # Save individual heatmaps
            heatmap_dir = save_dir / f"{sample_id}_{method_name}_heatmaps"
            self.grad_cam.save_heatmaps(heatmaps, heatmap_dir, f"{sample_id}")
        
        return heatmaps
    
    def batch_analyze(self, dataloader, num_samples: int = 10, save_dir: Optional[Path] = None) -> List[Dict]:
        """
        Analyze multiple samples from a dataloader.
        
        Args:
            dataloader: DataLoader containing samples
            num_samples: Number of samples to analyze
            save_dir: Directory to save results
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            mnv, fluid, ga, drusen, target, sample_ids = batch
            
            # Process first sample in batch
            inputs = [mnv[0:1], fluid[0:1], ga[0:1], drusen[0:1]]
            sample_id = sample_ids[0] if isinstance(sample_ids, list) else f"sample_{i}"
            true_class = target[0].item()
            
            # Predict class
            with torch.no_grad():
                pred = self.model(*inputs)
                pred_class = pred.argmax(dim=1).item()
            
            print(f"Analyzing sample {i+1}/{num_samples}: {sample_id}")
            print(f"  True class: {true_class}, Predicted class: {pred_class}")
            
            # Generate analysis
            sample_save_dir = save_dir / f"sample_{i:03d}_{sample_id}" if save_dir else None
            heatmaps = self.analyze_sample(inputs, pred_class, sample_save_dir, sample_id)
            
            results.append({
                'sample_id': sample_id,
                'true_class': true_class,
                'pred_class': pred_class,
                'heatmaps': heatmaps
            })
        
        print(f"Batch analysis completed: {len(results)} samples processed")
        return results
    
    def cleanup(self):
        """Clean up resources."""
        self.grad_cam.remove_hooks()


class ComparisonVisualizer:
    """
    Visualizer that compares Grad-CAM and Grad-CAM++ side by side.
    """
    
    def __init__(self, model, use_cuda: bool = True):
        self.model = model
        self.target_layers = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
        self.grad_cam = MultiInputGradCAM(model, self.target_layers, use_cuda)
        self.grad_cam_pp = MultiInputGradCAMPlusPlus(model, self.target_layers, use_cuda)
    
    def compare_methods(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                       save_path: Optional[Path] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate heatmaps using both Grad-CAM and Grad-CAM++ methods.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class index. If None, uses predicted class.
            save_path: Path to save the comparison visualization
            
        Returns:
            Tuple of (grad_cam_heatmaps, grad_cam_pp_heatmaps)
        """
        # Generate heatmaps with both methods
        grad_cam_heatmaps = self.grad_cam.generate_cam(inputs, target_class)
        grad_cam_pp_heatmaps = self.grad_cam_pp.generate_cam(inputs, target_class)
        
        # Create comparison visualization
        if save_path:
            self._create_comparison_plot(inputs, grad_cam_heatmaps, grad_cam_pp_heatmaps, 
                                       target_class, save_path)
        
        return grad_cam_heatmaps, grad_cam_pp_heatmaps
    
    def _create_comparison_plot(self, inputs: List[torch.Tensor], 
                               grad_cam_heatmaps: Dict[str, np.ndarray],
                               grad_cam_pp_heatmaps: Dict[str, np.ndarray],
                               target_class: Optional[int], save_path: Path) -> None:
        """Create a comparison plot showing both methods."""
        input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
        
        # Create figure with 3 rows: Original, Grad-CAM, Grad-CAM++
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
            # Original image
            img = inputs[i].squeeze().detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] <= 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Row 0: Original images
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f'{display_name} Input')
            axes[0, i].axis('off')
            
            # Prepare image for overlay
            if img.ndim == 2:  # Grayscale
                img_colored = plt.cm.get_cmap('gray')(img)[:, :, :3]
            else:
                img_colored = img
            
            # Row 1: Grad-CAM
            if inp_name in grad_cam_heatmaps:
                heatmap = grad_cam_heatmaps[inp_name]
                heatmap_colored = plt.cm.get_cmap('jet')(heatmap)[:, :, :3]
                overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{display_name} Grad-CAM')
            else:
                axes[1, i].text(0.5, 0.5, 'No heatmap\navailable', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].set_title(f'{display_name} Grad-CAM')
            axes[1, i].axis('off')
            
            # Row 2: Grad-CAM++
            if inp_name in grad_cam_pp_heatmaps:
                heatmap = grad_cam_pp_heatmaps[inp_name]
                heatmap_colored = plt.cm.get_cmap('jet')(heatmap)[:, :, :3]
                overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                axes[2, i].imshow(overlay)
                axes[2, i].set_title(f'{display_name} Grad-CAM++')
            else:
                axes[2, i].text(0.5, 0.5, 'No heatmap\navailable', 
                               transform=axes[2, i].transAxes, ha='center', va='center')
                axes[2, i].set_title(f'{display_name} Grad-CAM++')
            axes[2, i].axis('off')
        
        title = f'Grad-CAM vs Grad-CAM++ Comparison'
        if target_class is not None:
            title += f' - Target Class: {target_class}'
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to: {save_path}")
        plt.show()
    
    def analyze_sample_comparison(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                                 save_dir: Optional[Path] = None, sample_id: str = "sample") -> Dict:
        """
        Analyze a single sample with both methods and save comparison results.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class for analysis
            save_dir: Directory to save results
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary containing both sets of heatmaps and metadata
        """
        # Generate heatmaps with both methods
        grad_cam_heatmaps, grad_cam_pp_heatmaps = self.compare_methods(inputs, target_class)
        
        # Save results if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comparison visualization
            comparison_path = save_dir / f"{sample_id}_comparison.png"
            self._create_comparison_plot(inputs, grad_cam_heatmaps, grad_cam_pp_heatmaps, 
                                       target_class, comparison_path)
            
            # Save individual heatmaps for both methods
            gradcam_dir = save_dir / f"{sample_id}_gradcam_heatmaps"
            gradcampp_dir = save_dir / f"{sample_id}_gradcampp_heatmaps"
            
            self.grad_cam.save_heatmaps(grad_cam_heatmaps, gradcam_dir, f"{sample_id}_gradcam")
            self.grad_cam_pp.save_heatmaps(grad_cam_pp_heatmaps, gradcampp_dir, f"{sample_id}_gradcampp")
        
        return {
            'sample_id': sample_id,
            'target_class': target_class,
            'grad_cam_heatmaps': grad_cam_heatmaps,
            'grad_cam_pp_heatmaps': grad_cam_pp_heatmaps
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.grad_cam.remove_hooks()
        self.grad_cam_pp.remove_hooks()


class MultiInputSmoothGrad:
    """
    SmoothGrad implementation for multi-input models.
    
    SmoothGrad reduces noise in gradient-based visualizations by averaging
    gradients computed on multiple noisy versions of the input image.
    
    Reference: "SmoothGrad: removing noise by adding noise" (Smilkov et al., 2017)
    
    Args:
        model: The neural network model
        use_cuda: Whether to use CUDA if available
        n_samples: Number of noisy samples to generate (default: 50)
        noise_level: Standard deviation for Gaussian noise (default: 0.15)
        
    Example:
        >>> smooth_grad = MultiInputSmoothGrad(model)
        >>> heatmaps = smooth_grad.generate_smooth_grad([mnv, fluid, ga, drusen], target_class=1)
    """
    
    def __init__(self, model, use_cuda: bool = True, n_samples: int = 50, noise_level: float = 0.15):
        self.model = model
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.n_samples = n_samples
        self.noise_level = noise_level
    
    def generate_smooth_grad(self, inputs: List[torch.Tensor], target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate SmoothGrad visualizations for multiple inputs.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Dictionary mapping input names to their SmoothGrad visualizations
        """
        input_names = ['mnv', 'fluid', 'ga', 'drusen']
        
        # Ensure inputs are on the correct device
        inputs = [inp.to(self.device) for inp in inputs]
        
        # Get predicted class if not specified
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(*inputs)
                target_class = outputs.argmax(dim=1).item()
        
        # Initialize gradient accumulators
        smooth_grads = {name: torch.zeros_like(inp) for name, inp in zip(input_names, inputs)}
        
        # Generate noisy samples and compute gradients
        for _ in range(self.n_samples):
            # Add Gaussian noise to inputs
            noisy_inputs = []
            for inp in inputs:
                noise = torch.randn_like(inp) * self.noise_level
                noisy_input = inp + noise
                noisy_input.requires_grad_(True)
                noisy_input.retain_grad()  # Ensure gradients are retained for non-leaf tensors
                noisy_inputs.append(noisy_input)
            
            # Forward pass
            outputs = self.model(*noisy_inputs)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            class_score = outputs[:, target_class]
            class_score.backward(retain_graph=False)
            
            # Accumulate gradients
            for i, (name, noisy_inp) in enumerate(zip(input_names, noisy_inputs)):
                if noisy_inp.grad is not None:
                    smooth_grads[name] += noisy_inp.grad.detach()
        
        # Average the gradients and convert to visualizations
        smooth_grad_maps = {}
        for i, name in enumerate(input_names):
            # Average gradients
            avg_grad = smooth_grads[name] / self.n_samples
            
            # Convert to heatmap (take absolute value and average across channels)
            if avg_grad.dim() == 4:  # [B, C, H, W]
                heatmap = torch.abs(avg_grad).mean(dim=1).squeeze().cpu().numpy()  # [H, W]
            else:  # [B, H, W]
                heatmap = torch.abs(avg_grad).squeeze().cpu().numpy()  # [H, W]
            
            # Normalize
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / (heatmap.max() + 1e-8)
            
            # Resize to original input size if needed
            input_size = inputs[i].shape[-2:]  # [H, W]
            if heatmap.shape != input_size:
                if HAS_OPENCV:
                    import cv2
                    heatmap = cv2.resize(heatmap, (input_size[1], input_size[0]))
                else:
                    # Use PIL for resizing - handle edge cases
                    if heatmap.ndim == 0:  # Scalar case
                        heatmap = np.ones(input_size) * heatmap
                    elif heatmap.ndim == 1:  # 1D case
                        heatmap = np.tile(heatmap.reshape(-1, 1), (1, input_size[1]))
                        if heatmap.shape[0] != input_size[0]:
                            heatmap = np.tile(heatmap, (input_size[0], 1))
                    else:  # 2D case
                        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
                        heatmap_pil = heatmap_pil.resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
                        heatmap = np.array(heatmap_pil).astype(np.float32) / 255.0
            
            smooth_grad_maps[name] = heatmap
        
        return smooth_grad_maps
    
    def visualize_smooth_grad(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray],
                             save_path: Optional[Path] = None, target_class: Optional[int] = None) -> None:
        """
        Visualize the original images with their corresponding SmoothGrad heatmaps.
        
        Args:
            inputs: List of input tensors
            heatmaps: Dictionary of heatmaps from generate_smooth_grad()
            save_path: Path to save the visualization
            target_class: Target class for title
        """
        input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
            # Original image
            img = inputs[i].squeeze().detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] <= 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f'{display_name} Input')
            axes[0, i].axis('off')
            
            # SmoothGrad heatmap overlay
            if inp_name in heatmaps:
                heatmap = heatmaps[inp_name]
                
                # Create overlay
                if img.ndim == 2:  # Grayscale
                    img_colored = plt.cm.get_cmap('gray')(img)[:, :, :3]
                else:
                    img_colored = img
                
                heatmap_colored = plt.cm.get_cmap('jet')(heatmap)[:, :, :3]
                overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{display_name} SmoothGrad')
            else:
                axes[1, i].text(0.5, 0.5, 'No heatmap\navailable', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].set_title(f'{display_name} SmoothGrad')
            
            axes[1, i].axis('off')
        
        title = f'SmoothGrad Visualization (n_samples={self.n_samples}, noise={self.noise_level})'
        if target_class is not None:
            title += f' - Target Class: {target_class}'
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SmoothGrad visualization saved to: {save_path}")
        
        plt.show()
    
    def save_heatmaps(self, heatmaps: Dict[str, np.ndarray], save_dir: Path,
                      filename_prefix: str = "smoothgrad") -> None:
        """
        Save individual SmoothGrad heatmaps as images.
        
        Args:
            heatmaps: Dictionary of heatmaps
            save_dir: Directory to save heatmaps
            filename_prefix: Prefix for filenames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for inp_name, heatmap in heatmaps.items():
            # Convert to colored heatmap
            colored_heatmap = plt.cm.get_cmap('jet')(heatmap)
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Save
            filename = f"{filename_prefix}_{inp_name}.png"
            filepath = save_dir / filename
            
            if HAS_OPENCV:
                import cv2
                cv2.imwrite(str(filepath), cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
            else:
                # Use PIL to save
                heatmap_pil = Image.fromarray(colored_heatmap)
                heatmap_pil.save(str(filepath))
        
        print(f"Individual SmoothGrad heatmaps saved to: {save_dir}")


class MultiInputVarGrad:
    """
    VarGrad (Variance Grad) implementation for multi-input models.
    
    VarGrad computes the variance of gradients across multiple noisy samples
    to highlight regions where the model's decision is most sensitive to noise.
    This can reveal important but potentially unstable decision boundaries.
    
    Args:
        model: The neural network model
        use_cuda: Whether to use CUDA if available
        n_samples: Number of noisy samples to generate (default: 50)
        noise_level: Standard deviation for Gaussian noise (default: 0.15)
        
    Example:
        >>> var_grad = MultiInputVarGrad(model)
        >>> heatmaps = var_grad.generate_var_grad([mnv, fluid, ga, drusen], target_class=1)
    """
    
    def __init__(self, model, use_cuda: bool = True, n_samples: int = 50, noise_level: float = 0.15):
        self.model = model
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.n_samples = n_samples
        self.noise_level = noise_level
    
    def generate_var_grad(self, inputs: List[torch.Tensor], target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate VarGrad visualizations for multiple inputs.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class index. If None, uses predicted class.
            
        Returns:
            Dictionary mapping input names to their VarGrad visualizations
        """
        input_names = ['mnv', 'fluid', 'ga', 'drusen']
        
        # Ensure inputs are on the correct device
        inputs = [inp.to(self.device) for inp in inputs]
        
        # Get predicted class if not specified
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(*inputs)
                target_class = outputs.argmax(dim=1).item()
        
        # Store all gradients for variance computation
        all_gradients = {name: [] for name in input_names}
        
        # Generate noisy samples and compute gradients
        for _ in range(self.n_samples):
            # Add Gaussian noise to inputs
            noisy_inputs = []
            for inp in inputs:
                noise = torch.randn_like(inp) * self.noise_level
                noisy_input = inp + noise
                noisy_input.requires_grad_(True)
                noisy_input.retain_grad()  # Ensure gradients are retained for non-leaf tensors
                noisy_inputs.append(noisy_input)
            
            # Forward pass
            outputs = self.model(*noisy_inputs)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Backward pass for target class
            class_score = outputs[:, target_class]
            class_score.backward(retain_graph=False)
            
            # Store gradients for each input
            for i, (name, noisy_inp) in enumerate(zip(input_names, noisy_inputs)):
                if noisy_inp.grad is not None:
                    grad = noisy_inp.grad.detach().clone()
                    all_gradients[name].append(grad)
        
        # Compute variance of gradients and convert to visualizations
        var_grad_maps = {}
        for i, name in enumerate(input_names):
            if all_gradients[name]:
                # Stack gradients: [n_samples, B, C, H, W] or [n_samples, B, H, W]
                stacked_grads = torch.stack(all_gradients[name], dim=0)
                
                # Compute variance across samples
                grad_var = torch.var(stacked_grads, dim=0)  # [B, C, H, W] or [B, H, W]
                
                # Convert to heatmap (average across channels if needed)
                if grad_var.dim() == 4:  # [B, C, H, W]
                    heatmap = grad_var.mean(dim=1).squeeze().cpu().numpy()  # [H, W]
                else:  # [B, H, W]
                    heatmap = grad_var.squeeze().cpu().numpy()  # [H, W]
                
                # Normalize
                heatmap = heatmap - heatmap.min()
                heatmap = heatmap / (heatmap.max() + 1e-8)
                
                # Resize to original input size if needed
                input_size = inputs[i].shape[-2:]  # [H, W]
                if heatmap.shape != input_size:
                    if HAS_OPENCV:
                        import cv2
                        heatmap = cv2.resize(heatmap, (input_size[1], input_size[0]))
                    else:
                        # Use PIL for resizing - handle edge cases
                        if heatmap.ndim == 0:  # Scalar case
                            heatmap = np.ones(input_size) * heatmap
                        elif heatmap.ndim == 1:  # 1D case
                            heatmap = np.tile(heatmap.reshape(-1, 1), (1, input_size[1]))
                            if heatmap.shape[0] != input_size[0]:
                                heatmap = np.tile(heatmap, (input_size[0], 1))
                        else:  # 2D case
                            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
                            heatmap_pil = heatmap_pil.resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
                            heatmap = np.array(heatmap_pil).astype(np.float32) / 255.0
                
                var_grad_maps[name] = heatmap
            else:
                # Fallback: create empty heatmap
                input_size = inputs[i].shape[-2:]
                var_grad_maps[name] = np.zeros(input_size)
        
        return var_grad_maps
    
    def visualize_var_grad(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray],
                          save_path: Optional[Path] = None, target_class: Optional[int] = None) -> None:
        """
        Visualize the original images with their corresponding VarGrad heatmaps.
        
        Args:
            inputs: List of input tensors
            heatmaps: Dictionary of heatmaps from generate_var_grad()
            save_path: Path to save the visualization
            target_class: Target class for title
        """
        input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
            # Original image
            img = inputs[i].squeeze().detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] <= 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f'{display_name} Input')
            axes[0, i].axis('off')
            
            # VarGrad heatmap overlay
            if inp_name in heatmaps:
                heatmap = heatmaps[inp_name]
                
                # Create overlay
                if img.ndim == 2:  # Grayscale
                    img_colored = plt.cm.get_cmap('gray')(img)[:, :, :3]
                else:
                    img_colored = img
                
                heatmap_colored = plt.cm.get_cmap('jet')(heatmap)[:, :, :3]  # Using jet colormap for consistency
                overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{display_name} VarGrad')
            else:
                axes[1, i].text(0.5, 0.5, 'No heatmap\navailable', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].set_title(f'{display_name} VarGrad')
            
            axes[1, i].axis('off')
        
        title = f'VarGrad Visualization (n_samples={self.n_samples}, noise={self.noise_level})'
        if target_class is not None:
            title += f' - Target Class: {target_class}'
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"VarGrad visualization saved to: {save_path}")
        
        plt.show()
    
    def save_heatmaps(self, heatmaps: Dict[str, np.ndarray], save_dir: Path,
                      filename_prefix: str = "vargrad") -> None:
        """
        Save individual VarGrad heatmaps as images.
        
        Args:
            heatmaps: Dictionary of heatmaps
            save_dir: Directory to save heatmaps
            filename_prefix: Prefix for filenames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for inp_name, heatmap in heatmaps.items():
            # Convert to colored heatmap
            colored_heatmap = plt.cm.get_cmap('jet')(heatmap)
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            
            # Save
            filename = f"{filename_prefix}_{inp_name}.png"
            filepath = save_dir / filename
            
            if HAS_OPENCV:
                import cv2
                cv2.imwrite(str(filepath), cv2.cvtColor(colored_heatmap, cv2.COLOR_RGB2BGR))
            else:
                # Use PIL to save
                heatmap_pil = Image.fromarray(colored_heatmap)
                heatmap_pil.save(str(filepath))
        
        print(f"Individual VarGrad heatmaps saved to: {save_dir}")


class SmoothGradVisualizer:
    """
    High-level interface for SmoothGrad visualization.
    """
    
    def __init__(self, model, use_cuda: bool = True, n_samples: int = 50, noise_level: float = 0.15):
        self.model = model
        self.smooth_grad = MultiInputSmoothGrad(model, use_cuda, n_samples, noise_level)
    
    def analyze_sample(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                      save_dir: Optional[Path] = None, sample_id: str = "sample") -> Dict[str, np.ndarray]:
        """
        Analyze a single sample and generate SmoothGrad visualizations.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class for analysis
            save_dir: Directory to save results
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary of heatmaps
        """
        # Generate heatmaps
        heatmaps = self.smooth_grad.generate_smooth_grad(inputs, target_class)
        
        # Save results if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            viz_path = save_dir / f"{sample_id}_smoothgrad_visualization.png"
            self.smooth_grad.visualize_smooth_grad(inputs, heatmaps, viz_path, target_class)
            
            # Save individual heatmaps
            heatmap_dir = save_dir / f"{sample_id}_smoothgrad_heatmaps"
            self.smooth_grad.save_heatmaps(heatmaps, heatmap_dir, f"{sample_id}")
        
        return heatmaps
    
    def cleanup(self):
        """Clean up resources."""
        pass  # SmoothGrad doesn't use hooks


class VarGradVisualizer:
    """
    High-level interface for VarGrad visualization.
    """
    
    def __init__(self, model, use_cuda: bool = True, n_samples: int = 50, noise_level: float = 0.15):
        self.model = model
        self.var_grad = MultiInputVarGrad(model, use_cuda, n_samples, noise_level)
    
    def analyze_sample(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                      save_dir: Optional[Path] = None, sample_id: str = "sample") -> Dict[str, np.ndarray]:
        """
        Analyze a single sample and generate VarGrad visualizations.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class for analysis
            save_dir: Directory to save results
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary of heatmaps
        """
        # Generate heatmaps
        heatmaps = self.var_grad.generate_var_grad(inputs, target_class)
        
        # Save results if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualization
            viz_path = save_dir / f"{sample_id}_vargrad_visualization.png"
            self.var_grad.visualize_var_grad(inputs, heatmaps, viz_path, target_class)
            
            # Save individual heatmaps
            heatmap_dir = save_dir / f"{sample_id}_vargrad_heatmaps"
            self.var_grad.save_heatmaps(heatmaps, heatmap_dir, f"{sample_id}")
        
        return heatmaps
    
    def cleanup(self):
        """Clean up resources."""
        pass  # VarGrad doesn't use hooks


class AllMethodsComparison:
    """
    Comprehensive visualizer that compares all methods: Grad-CAM, Grad-CAM++, SmoothGrad, and VarGrad.
    """
    
    def __init__(self, model, use_cuda: bool = True, n_samples: int = 50, noise_level: float = 0.15):
        self.model = model
        self.target_layers = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
        self.grad_cam = MultiInputGradCAM(model, self.target_layers, use_cuda)
        self.grad_cam_pp = MultiInputGradCAMPlusPlus(model, self.target_layers, use_cuda)
        self.smooth_grad = MultiInputSmoothGrad(model, use_cuda, n_samples, noise_level)
        self.var_grad = MultiInputVarGrad(model, use_cuda, n_samples, noise_level)
    
    def compare_all_methods(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                           save_path: Optional[Path] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate heatmaps using all four methods.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class index. If None, uses predicted class.
            save_path: Path to save the comparison visualization
            
        Returns:
            Dictionary containing all method results
        """
        # Generate heatmaps with all methods
        grad_cam_heatmaps = self.grad_cam.generate_cam(inputs, target_class)
        grad_cam_pp_heatmaps = self.grad_cam_pp.generate_cam(inputs, target_class)
        smooth_grad_heatmaps = self.smooth_grad.generate_smooth_grad(inputs, target_class)
        var_grad_heatmaps = self.var_grad.generate_var_grad(inputs, target_class)
        
        # Create comprehensive comparison visualization
        if save_path:
            self._create_all_methods_plot(inputs, {
                'gradcam': grad_cam_heatmaps,
                'gradcam++': grad_cam_pp_heatmaps,
                'smoothgrad': smooth_grad_heatmaps,
                'vargrad': var_grad_heatmaps
            }, target_class, save_path)
        
        return {
            'gradcam': grad_cam_heatmaps,
            'gradcam++': grad_cam_pp_heatmaps,
            'smoothgrad': smooth_grad_heatmaps,
            'vargrad': var_grad_heatmaps
        }
    
    def _create_all_methods_plot(self, inputs: List[torch.Tensor], 
                                all_heatmaps: Dict[str, Dict[str, np.ndarray]],
                                target_class: Optional[int], save_path: Path) -> None:
        """Create a comprehensive comparison plot showing all methods."""
        input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
        method_names = ['Original', 'Grad-CAM', 'Grad-CAM++', 'SmoothGrad', 'VarGrad']
        colormaps = ['gray', 'jet', 'jet', 'hot', 'plasma']
        
        # Create figure with 5 rows: Original, Grad-CAM, Grad-CAM++, SmoothGrad, VarGrad
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        
        for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
            # Original image
            img = inputs[i].squeeze().detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] <= 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 3 and img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            
            # Normalize for display
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Row 0: Original images
            axes[0, i].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, i].set_title(f'{display_name} Input')
            axes[0, i].axis('off')
            
            # Prepare image for overlay
            if img.ndim == 2:  # Grayscale
                img_colored = plt.cm.get_cmap('gray')(img)[:, :, :3]
            else:
                img_colored = img
            
            # Rows 1-4: Different methods
            method_keys = ['gradcam', 'gradcam++', 'smoothgrad', 'vargrad']
            method_cmaps = ['jet', 'jet', 'jet', 'jet']  # Standardized to jet colormap
            
            for row, (method_key, method_cmap) in enumerate(zip(method_keys, method_cmaps), 1):
                if method_key in all_heatmaps and inp_name in all_heatmaps[method_key]:
                    heatmap = all_heatmaps[method_key][inp_name]
                    heatmap_colored = plt.cm.get_cmap(method_cmap)(heatmap)[:, :, :3]
                    overlay = 0.6 * img_colored + 0.4 * heatmap_colored
                    axes[row, i].imshow(overlay)
                    axes[row, i].set_title(f'{display_name} {method_names[row]}')
                else:
                    axes[row, i].text(0.5, 0.5, 'No heatmap\navailable', 
                                     transform=axes[row, i].transAxes, ha='center', va='center')
                    axes[row, i].set_title(f'{display_name} {method_names[row]}')
                axes[row, i].axis('off')
        
        title = f'All Methods Comparison'
        if target_class is not None:
            title += f' - Target Class: {target_class}'
        
        plt.suptitle(title, fontsize=18)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All methods comparison saved to: {save_path}")
        plt.show()
    
    def analyze_sample_all_methods(self, inputs: List[torch.Tensor], target_class: Optional[int] = None,
                                  save_dir: Optional[Path] = None, sample_id: str = "sample") -> Dict:
        """
        Analyze a single sample with all methods and save results.
        
        Args:
            inputs: List of input tensors [mnv, fluid, ga, drusen]
            target_class: Target class for analysis
            save_dir: Directory to save results
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary containing all method results and metadata
        """
        # Generate heatmaps with all methods
        all_results = self.compare_all_methods(inputs, target_class)
        
        # Save results if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comprehensive comparison visualization
            comparison_path = save_dir / f"{sample_id}_all_methods_comparison.png"
            self._create_all_methods_plot(inputs, all_results, target_class, comparison_path)
            
            # Save individual heatmaps for all methods
            for method_name, heatmaps in all_results.items():
                method_dir = save_dir / f"{sample_id}_{method_name}_heatmaps"
                
                if method_name == 'gradcam':
                    self.grad_cam.save_heatmaps(heatmaps, method_dir, f"{sample_id}_{method_name}")
                elif method_name == 'gradcam++':
                    self.grad_cam_pp.save_heatmaps(heatmaps, method_dir, f"{sample_id}_{method_name}")
                elif method_name == 'smoothgrad':
                    self.smooth_grad.save_heatmaps(heatmaps, method_dir, f"{sample_id}_{method_name}")
                elif method_name == 'vargrad':
                    self.var_grad.save_heatmaps(heatmaps, method_dir, f"{sample_id}_{method_name}")
        
        return {
            'sample_id': sample_id,
            'target_class': target_class,
            'all_methods': all_results
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.grad_cam.remove_hooks()
        self.grad_cam_pp.remove_hooks()
        # SmoothGrad and VarGrad don't use hooks