#!/usr/bin/env python3
"""
Simple Grad-CAM++ Demo

This script demonstrates the key differences between Grad-CAM and Grad-CAM++
by creating synthetic examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Utils.grad_cam import MultiInputGradCAM, MultiInputGradCAMPlusPlus


class SimpleTestModel(nn.Module):
    """Simple 4-branch model for testing."""
    
    def __init__(self):
        super().__init__()
        # Four simple backbone networks
        self.backbone1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.backbone2 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.backbone3 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.backbone4 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Linear(32 * 4, 4)  # 4 classes
    
    def forward(self, x1, x2, x3, x4):
        f1 = self.backbone1(x1).flatten(1)
        f2 = self.backbone2(x2).flatten(1)
        f3 = self.backbone3(x3).flatten(1)
        f4 = self.backbone4(x4).flatten(1)
        
        combined = torch.cat([f1, f2, f3, f4], dim=1)
        return self.classifier(combined)


def demo_gradcam_plusplus():
    """Demonstrate Grad-CAM++ functionality."""
    print("🎯 Grad-CAM++ Demo")
    print("=" * 40)
    
    # Create test model
    print("🔧 Creating test model...")
    model = SimpleTestModel()
    model.eval()
    
    # Create synthetic inputs
    print("🎨 Generating synthetic inputs...")
    batch_size = 1
    height, width = 64, 64
    
    # Create test inputs with different patterns
    x1 = torch.randn(batch_size, 1, height, width) * 0.5 + 0.5  # MNV
    x2 = torch.randn(batch_size, 1, height, width) * 0.3 + 0.3  # Fluid
    x3 = torch.randn(batch_size, 1, height, width) * 0.4 + 0.2  # GA
    x4 = torch.randn(batch_size, 1, height, width) * 0.6 + 0.1  # Drusen
    
    inputs = [x1, x2, x3, x4]
    input_names = ['MNV', 'Fluid', 'GA', 'Drusen']
    
    # Make prediction
    print("🔮 Making prediction...")
    with torch.no_grad():
        logits = model(*inputs)
        predicted_class = int(logits.argmax(dim=1).item())
        probabilities = F.softmax(logits, dim=1)
        confidence = float(probabilities[0, predicted_class].item())
    
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {confidence:.3f}")
    
    # Test regular Grad-CAM
    print("\n📊 Testing regular Grad-CAM...")
    target_layers = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
    
    gradcam = MultiInputGradCAM(model, target_layers, use_cuda=False)
    gradcam_heatmaps = gradcam.generate_cam(inputs, target_class=predicted_class)
    gradcam.remove_hooks()
    
    print(f"   Generated {len(gradcam_heatmaps)} heatmaps")
    
    # Test Grad-CAM++
    print("\n✨ Testing Grad-CAM++...")
    gradcam_pp = MultiInputGradCAMPlusPlus(model, target_layers, use_cuda=False)
    gradcam_pp_heatmaps = gradcam_pp.generate_cam(inputs, target_class=predicted_class)
    gradcam_pp.remove_hooks()
    
    print(f"   Generated {len(gradcam_pp_heatmaps)} heatmaps")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    output_dir = Path("gradcam_plusplus_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, (inp_name, display_name) in enumerate(zip(['mnv', 'fluid', 'ga', 'drusen'], input_names)):
        # Original image
        img = inputs[i].squeeze().detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'{display_name} Input')
        axes[0, i].axis('off')
        
        # Grad-CAM heatmap
        if inp_name in gradcam_heatmaps:
            heatmap = gradcam_heatmaps[inp_name]
            overlay = 0.6 * img + 0.4 * heatmap
            axes[1, i].imshow(overlay, cmap='hot')
            axes[1, i].set_title(f'{display_name} Grad-CAM')
        else:
            axes[1, i].text(0.5, 0.5, 'No heatmap', transform=axes[1, i].transAxes, ha='center')
            axes[1, i].set_title(f'{display_name} Grad-CAM')
        axes[1, i].axis('off')
        
        # Grad-CAM++ heatmap
        if inp_name in gradcam_pp_heatmaps:
            heatmap = gradcam_pp_heatmaps[inp_name]
            overlay = 0.6 * img + 0.4 * heatmap
            axes[2, i].imshow(overlay, cmap='hot')
            axes[2, i].set_title(f'{display_name} Grad-CAM++')
        else:
            axes[2, i].text(0.5, 0.5, 'No heatmap', transform=axes[2, i].transAxes, ha='center')
            axes[2, i].set_title(f'{display_name} Grad-CAM++')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Grad-CAM vs Grad-CAM++ Comparison\nPredicted Class: {predicted_class}, Confidence: {confidence:.3f}', fontsize=16)
    plt.tight_layout()
    
    demo_path = output_dir / "gradcam_plusplus_demo.png"
    plt.savefig(demo_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to: {demo_path}")
    
    # Analyze differences
    print("\n📈 Analyzing differences...")
    for inp_name in ['mnv', 'fluid', 'ga', 'drusen']:
        if inp_name in gradcam_heatmaps and inp_name in gradcam_pp_heatmaps:
            gc_map = gradcam_heatmaps[inp_name]
            gcp_map = gradcam_pp_heatmaps[inp_name]
            
            gc_max = gc_map.max()
            gcp_max = gcp_map.max()
            gc_mean = gc_map.mean()
            gcp_mean = gcp_map.mean()
            
            print(f"   {inp_name.upper()}:")
            print(f"     Grad-CAM   - Max: {gc_max:.3f}, Mean: {gc_mean:.3f}")
            print(f"     Grad-CAM++ - Max: {gcp_max:.3f}, Mean: {gcp_mean:.3f}")
            print(f"     Difference - Max: {abs(gcp_max - gc_max):.3f}, Mean: {abs(gcp_mean - gc_mean):.3f}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   Check the {output_dir} folder for visualization")
    
    # Key differences summary
    print("\n📋 Key Differences:")
    print("   🔸 Grad-CAM: Uses global average pooling of gradients")
    print("   🔸 Grad-CAM++: Uses pixel-wise weighting for better localization")
    print("   🔸 Grad-CAM++: Better at handling multiple objects/complex scenes")
    print("   🔸 Grad-CAM++: More computationally intensive but more accurate")


if __name__ == "__main__":
    demo_gradcam_plusplus()