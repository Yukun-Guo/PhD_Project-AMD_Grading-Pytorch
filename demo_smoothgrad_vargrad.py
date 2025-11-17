"""
Simple demo of SmoothGrad and VarGrad implementations.

This script demonstrates the new gradient-based visualization methods
using synthetic data to avoid dependencies on the full data pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import toml
from NetModule import NetModule
from Utils.grad_cam import (
    MultiInputSmoothGrad, MultiInputVarGrad, 
    SmoothGradVisualizer, VarGradVisualizer,
    AllMethodsComparison
)


def create_synthetic_data():
    """Create synthetic multi-input data for testing."""
    print("Creating synthetic test data...")
    
    # Create synthetic images with different patterns
    batch_size = 1
    height, width = 224, 224
    
    # MNV: Diagonal pattern
    mnv = torch.zeros(batch_size, 3, height, width)
    for i in range(height):
        for j in range(width):
            if abs(i - j) < 20:  # Diagonal band
                mnv[0, :, i, j] = 0.8
    
    # Fluid: Circular pattern
    fluid = torch.zeros(batch_size, 3, height, width)
    center_x, center_y = height // 2, width // 2
    radius = 50
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < radius:
                fluid[0, :, i, j] = 0.9
    
    # GA: Grid pattern
    ga = torch.zeros(batch_size, 3, height, width)
    for i in range(0, height, 30):
        ga[0, :, i:i+10, :] = 0.7
    for j in range(0, width, 30):
        ga[0, :, :, j:j+10] = 0.7
    
    # Drusen: Random spots
    drusen = torch.zeros(batch_size, 3, height, width)
    np.random.seed(42)  # For reproducibility
    for _ in range(20):
        x = np.random.randint(0, height-20)
        y = np.random.randint(0, width-20)
        drusen[0, :, x:x+20, y:y+20] = np.random.uniform(0.5, 1.0)
    
    # Add some noise
    mnv += torch.randn_like(mnv) * 0.1
    fluid += torch.randn_like(fluid) * 0.1
    ga += torch.randn_like(ga) * 0.1
    drusen += torch.randn_like(drusen) * 0.1
    
    # Clamp to valid range
    mnv = torch.clamp(mnv, 0, 1)
    fluid = torch.clamp(fluid, 0, 1)
    ga = torch.clamp(ga, 0, 1)
    drusen = torch.clamp(drusen, 0, 1)
    
    return [mnv, fluid, ga, drusen]


def create_simple_model():
    """Create a simple model for testing without full configuration."""
    print("Creating simple test model...")
    
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Simple backbone for each input
            self.backbone1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
            self.backbone2 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
            self.backbone3 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
            self.backbone4 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
            
            # Classifier
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 4)  # 4 classes
            )
        
        def forward(self, mnv, fluid, ga, drusen):
            f1 = self.backbone1(mnv)
            f2 = self.backbone2(fluid)
            f3 = self.backbone3(ga)
            f4 = self.backbone4(drusen)
            
            combined = torch.cat([f1, f2, f3, f4], dim=1)
            return self.classifier(combined)
    
    model = SimpleTestModel()
    model.eval()
    return model


def demo_smoothgrad():
    """Demonstrate SmoothGrad functionality."""
    print("\n" + "=" * 50)
    print("DEMONSTRATING SMOOTHGRAD")
    print("=" * 50)
    
    # Create test data and model
    inputs = create_synthetic_data()
    model = create_simple_model()
    
    # Create output directory
    output_dir = Path("demo_smoothgrad")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating SmoothGrad visualizer...")
    smooth_grad = MultiInputSmoothGrad(model, use_cuda=False, n_samples=30, noise_level=0.15)
    
    print("Generating SmoothGrad heatmaps...")
    heatmaps = smooth_grad.generate_smooth_grad(inputs, target_class=1)
    
    print("Creating visualization...")
    smooth_grad.visualize_smooth_grad(inputs, heatmaps, 
                                     save_path=output_dir / "smoothgrad_demo.png",
                                     target_class=1)
    
    # Save individual heatmaps
    smooth_grad.save_heatmaps(heatmaps, output_dir, "smoothgrad_demo")
    
    print(f"✓ SmoothGrad demo completed. Results saved to: {output_dir}")
    
    # Print statistics
    print("Heatmap statistics:")
    for inp_name, heatmap in heatmaps.items():
        print(f"  {inp_name}: shape={heatmap.shape}, max={heatmap.max():.4f}, "
              f"mean={heatmap.mean():.4f}, std={heatmap.std():.4f}")


def demo_vargrad():
    """Demonstrate VarGrad functionality."""
    print("\n" + "=" * 50)
    print("DEMONSTRATING VARGRAD")
    print("=" * 50)
    
    # Create test data and model
    inputs = create_synthetic_data()
    model = create_simple_model()
    
    # Create output directory
    output_dir = Path("demo_vargrad")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating VarGrad visualizer...")
    var_grad = MultiInputVarGrad(model, use_cuda=False, n_samples=30, noise_level=0.15)
    
    print("Generating VarGrad heatmaps...")
    heatmaps = var_grad.generate_var_grad(inputs, target_class=1)
    
    print("Creating visualization...")
    var_grad.visualize_var_grad(inputs, heatmaps, 
                               save_path=output_dir / "vargrad_demo.png",
                               target_class=1)
    
    # Save individual heatmaps
    var_grad.save_heatmaps(heatmaps, output_dir, "vargrad_demo")
    
    print(f"✓ VarGrad demo completed. Results saved to: {output_dir}")
    
    # Print statistics
    print("Heatmap statistics:")
    for inp_name, heatmap in heatmaps.items():
        print(f"  {inp_name}: shape={heatmap.shape}, max={heatmap.max():.4f}, "
              f"mean={heatmap.mean():.4f}, std={heatmap.std():.4f}")


def demo_all_methods_comparison():
    """Demonstrate all methods comparison."""
    print("\n" + "=" * 50)
    print("DEMONSTRATING ALL METHODS COMPARISON")
    print("=" * 50)
    
    # Create test data and model
    inputs = create_synthetic_data()
    model = create_simple_model()
    
    # Create output directory
    output_dir = Path("demo_all_methods")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating all methods comparison...")
    all_methods = AllMethodsComparison(model, use_cuda=False, n_samples=20, noise_level=0.1)
    
    print("Generating comparison for all methods...")
    comparison_path = output_dir / "all_methods_comparison.png"
    all_results = all_methods.compare_all_methods(inputs, target_class=1, save_path=comparison_path)
    
    all_methods.cleanup()
    
    print(f"✓ All methods comparison completed. Results saved to: {output_dir}")
    
    # Print comparison statistics
    print("Method comparison statistics:")
    for method_name, heatmaps in all_results.items():
        print(f"\n  {method_name.upper()}:")
        for inp_name, heatmap in heatmaps.items():
            print(f"    {inp_name}: max={heatmap.max():.4f}, mean={heatmap.mean():.4f}")


def demo_high_level_visualizers():
    """Demonstrate high-level SmoothGradVisualizer and VarGradVisualizer."""
    print("\n" + "=" * 50)
    print("DEMONSTRATING HIGH-LEVEL VISUALIZERS")
    print("=" * 50)
    
    # Create test data and model
    inputs = create_synthetic_data()
    model = create_simple_model()
    
    # Create output directory
    output_dir = Path("demo_high_level")
    output_dir.mkdir(exist_ok=True)
    
    # Test SmoothGradVisualizer
    print("Testing SmoothGradVisualizer...")
    smooth_viz = SmoothGradVisualizer(model, use_cuda=False, n_samples=25, noise_level=0.12)
    smooth_results = smooth_viz.analyze_sample(inputs, target_class=2, 
                                             save_dir=output_dir, sample_id="high_level_smooth")
    smooth_viz.cleanup()
    
    # Test VarGradVisualizer
    print("Testing VarGradVisualizer...")
    var_viz = VarGradVisualizer(model, use_cuda=False, n_samples=25, noise_level=0.12)
    var_results = var_viz.analyze_sample(inputs, target_class=2,
                                       save_dir=output_dir, sample_id="high_level_var")
    var_viz.cleanup()
    
    print(f"✓ High-level visualizers demo completed. Results saved to: {output_dir}")


def create_parameter_sensitivity_analysis():
    """Analyze sensitivity to different parameters."""
    print("\n" + "=" * 50)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    # Create test data and model
    inputs = create_synthetic_data()
    model = create_simple_model()
    
    # Test different parameter combinations
    n_samples_values = [10, 25, 50]
    noise_levels = [0.05, 0.15, 0.25]
    
    results = {}
    
    print("Testing SmoothGrad parameter sensitivity...")
    for n_samples in n_samples_values:
        for noise_level in noise_levels:
            print(f"  Testing n_samples={n_samples}, noise_level={noise_level}")
            
            smooth_grad = MultiInputSmoothGrad(model, use_cuda=False, 
                                             n_samples=n_samples, noise_level=noise_level)
            heatmaps = smooth_grad.generate_smooth_grad(inputs, target_class=1)
            
            # Calculate average statistics
            avg_max = np.mean([hm.max() for hm in heatmaps.values()])
            avg_mean = np.mean([hm.mean() for hm in heatmaps.values()])
            avg_std = np.mean([hm.std() for hm in heatmaps.values()])
            
            results[f"smooth_{n_samples}_{noise_level}"] = {
                'avg_max': avg_max, 'avg_mean': avg_mean, 'avg_std': avg_std
            }
    
    print("Testing VarGrad parameter sensitivity...")
    for n_samples in n_samples_values:
        for noise_level in noise_levels:
            print(f"  Testing n_samples={n_samples}, noise_level={noise_level}")
            
            var_grad = MultiInputVarGrad(model, use_cuda=False, 
                                       n_samples=n_samples, noise_level=noise_level)
            heatmaps = var_grad.generate_var_grad(inputs, target_class=1)
            
            # Calculate average statistics
            avg_max = np.mean([hm.max() for hm in heatmaps.values()])
            avg_mean = np.mean([hm.mean() for hm in heatmaps.values()])
            avg_std = np.mean([hm.std() for hm in heatmaps.values()])
            
            results[f"var_{n_samples}_{noise_level}"] = {
                'avg_max': avg_max, 'avg_mean': avg_mean, 'avg_std': avg_std
            }
    
    # Create visualization
    output_dir = Path("demo_parameter_sensitivity")
    output_dir.mkdir(exist_ok=True)
    
    # Print results
    print("\nParameter sensitivity results:")
    for key, stats in results.items():
        print(f"  {key}: max={stats['avg_max']:.4f}, mean={stats['avg_mean']:.4f}, std={stats['avg_std']:.4f}")
    
    print(f"✓ Parameter sensitivity analysis completed.")


def main():
    """Run all demonstrations."""
    try:
        print("🚀 Starting SmoothGrad and VarGrad demonstration...")
        print("This demo uses synthetic data to showcase the new gradient visualization methods.")
        
        # Demo 1: SmoothGrad
        demo_smoothgrad()
        
        # Demo 2: VarGrad
        demo_vargrad()
        
        # Demo 3: All methods comparison
        demo_all_methods_comparison()
        
        # Demo 4: High-level visualizers
        demo_high_level_visualizers()
        
        # Demo 5: Parameter sensitivity
        create_parameter_sensitivity_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Results are saved in the following directories:")
        print("  - demo_smoothgrad/")
        print("  - demo_vargrad/")
        print("  - demo_all_methods/")
        print("  - demo_high_level/")
        print("\nGenerated files include:")
        print("  • Individual heatmap images for each input type")
        print("  • Combined visualization plots")
        print("  • All methods comparison plot")
        print("  • Parameter sensitivity analysis")
        
        print("\nKey features demonstrated:")
        print("  ✓ SmoothGrad: Noise-based gradient averaging for smoother visualizations")
        print("  ✓ VarGrad: Gradient variance analysis for uncertainty visualization")
        print("  ✓ All methods comparison: Side-by-side comparison of all 4 methods")
        print("  ✓ High-level interfaces: Easy-to-use visualizer classes")
        print("  ✓ Parameter sensitivity: Analysis of n_samples and noise_level effects")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()