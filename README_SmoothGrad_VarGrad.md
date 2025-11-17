# SmoothGrad and VarGrad Implementation

This document describes the newly implemented SmoothGrad and VarGrad gradient-based visualization methods for the AMD grading project.

## Overview

We have extended the existing Grad-CAM visualization capabilities with two additional methods:

1. **SmoothGrad**: Reduces noise in gradient-based visualizations by averaging gradients computed on multiple noisy versions of the input image
2. **VarGrad**: Computes the variance of gradients across multiple noisy samples to highlight regions where the model's decision is most sensitive to noise

## New Files and Components

### Core Implementation (Utils/grad_cam.py)

New classes added:
- `MultiInputSmoothGrad`: Core SmoothGrad implementation for multi-input models
- `MultiInputVarGrad`: Core VarGrad implementation for multi-input models  
- `SmoothGradVisualizer`: High-level interface for SmoothGrad analysis
- `VarGradVisualizer`: High-level interface for VarGrad analysis
- `AllMethodsComparison`: Comprehensive comparison of all four methods

### Integration (NetModule.py)

New methods added:
- `analyze_prediction_with_smoothgrad()`: NetModule integration for SmoothGrad
- `analyze_prediction_with_vargrad()`: NetModule integration for VarGrad
- `compare_all_gradient_methods()`: Compare all four gradient methods
- Updated `create_grad_cam_visualizer()` to support new methods

### Dataset Analysis (ModelGradCAM.py)

Enhanced to support:
- All four visualization methods: gradcam, gradcam++, smoothgrad, vargrad
- Method comparison modes: both (grad-cam + grad-cam++), all (all 4 methods)
- Configurable parameters: --n_samples, --noise_level
- Organized output structure for all methods

## Usage Examples

### Command Line (ModelGradCAM.py)

```bash
# SmoothGrad with custom parameters
python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method smoothgrad --n_samples 50 --noise_level 0.15

# VarGrad with custom parameters  
python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method vargrad --n_samples 30 --noise_level 0.1

# Compare all methods
python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method all --n_samples 25

# Traditional methods still work
python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method gradcam
python ModelGradCAM.py --config configs/config_bio.toml --dataset val --method both
```

### Programmatic Usage

```python
from NetModule import NetModule
from Utils.grad_cam import SmoothGradVisualizer, VarGradVisualizer, AllMethodsComparison

# Load model
model = NetModule(config=config)

# SmoothGrad analysis
smooth_results = model.analyze_prediction_with_smoothgrad(
    mnv, fluid, ga, drusen,
    target_class=1, 
    save_dir="output/",
    n_samples=50,
    noise_level=0.15
)

# VarGrad analysis
var_results = model.analyze_prediction_with_vargrad(
    mnv, fluid, ga, drusen,
    target_class=1,
    save_dir="output/",
    n_samples=30,
    noise_level=0.1
)

# All methods comparison
all_results = model.compare_all_gradient_methods(
    mnv, fluid, ga, drusen,
    target_class=1,
    save_dir="output/",
    n_samples=25
)
```

### Low-level API

```python
from Utils.grad_cam import MultiInputSmoothGrad, MultiInputVarGrad

# Direct SmoothGrad usage
smooth_grad = MultiInputSmoothGrad(model, use_cuda=True, n_samples=50, noise_level=0.15)
heatmaps = smooth_grad.generate_smooth_grad(inputs, target_class=1)
smooth_grad.visualize_smooth_grad(inputs, heatmaps, save_path="smoothgrad.png")

# Direct VarGrad usage
var_grad = MultiInputVarGrad(model, use_cuda=True, n_samples=50, noise_level=0.15)
heatmaps = var_grad.generate_var_grad(inputs, target_class=1)
var_grad.visualize_var_grad(inputs, heatmaps, save_path="vargrad.png")
```

## Parameters

### SmoothGrad & VarGrad Parameters

- `n_samples` (int): Number of noisy samples to generate (default: 50)
  - Higher values: More stable results, longer computation time
  - Lower values: Faster computation, potentially noisier results
  - Recommended range: 20-100

- `noise_level` (float): Standard deviation for Gaussian noise (default: 0.15)
  - Higher values: More aggressive noise, potentially stronger signal
  - Lower values: Subtle noise, cleaner but potentially weaker signal
  - Recommended range: 0.05-0.25

### Method Selection (--method parameter)

- `gradcam`: Traditional Grad-CAM
- `gradcam++`: Improved Grad-CAM with pixel-wise weighting
- `smoothgrad`: SmoothGrad with noise-based averaging
- `vargrad`: VarGrad with gradient variance analysis
- `both`: Comparison of Grad-CAM and Grad-CAM++
- `all`: Comprehensive comparison of all four methods

## Output Structure

When using ModelGradCAM.py, results are organized as:

```
CAMVis/
└── session_YYYYMMDD_HHMMSS/
    ├── heatmaps/
    │   ├── val/
    │   │   └── class_0_normal/
    │   │       ├── sample_001_smoothgrad/
    │   │       │   ├── sample_001_smoothgrad_mnv.png
    │   │       │   ├── sample_001_smoothgrad_fluid.png
    │   │       │   ├── sample_001_smoothgrad_ga.png
    │   │       │   └── sample_001_smoothgrad_drusen.png
    │   │       └── sample_001_vargrad/
    │   │           └── ...
    ├── visualizations/
    │   └── val/
    │       ├── sample_001_smoothgrad_visualization.png
    │       ├── sample_001_vargrad_visualization.png
    │       └── sample_001_all_methods_comparison.png
    └── reports/
        ├── overall_statistics.json
        └── detailed_analysis_report.txt
```

## Testing and Demos

### Demo Script
Run the standalone demo with synthetic data:
```bash
python demo_smoothgrad_vargrad.py
```

This creates:
- Individual method demonstrations
- Parameter sensitivity analysis
- All methods comparison
- High-level visualizer examples

### Integration Test
For testing with real data pipeline:
```bash
python test_smoothgrad_vargrad.py
```

## Key Features

### SmoothGrad
- **Purpose**: Reduce noise in gradient visualizations
- **Method**: Average gradients across multiple noisy input versions
- **Benefits**: Cleaner, more stable visualizations
- **Use case**: When gradient visualizations are too noisy or inconsistent

### VarGrad  
- **Purpose**: Analyze gradient uncertainty and sensitivity
- **Method**: Compute variance of gradients across noisy samples
- **Benefits**: Reveals regions of high decision uncertainty
- **Use case**: Understanding model confidence and decision boundaries

### All Methods Comparison
- **Purpose**: Side-by-side comparison of all visualization methods
- **Methods**: Grad-CAM, Grad-CAM++, SmoothGrad, VarGrad
- **Benefits**: Comprehensive analysis in single operation
- **Use case**: Research analysis and method comparison

## Technical Details

### Memory Usage
- SmoothGrad/VarGrad require more memory due to multiple forward passes
- Memory usage scales with `n_samples` parameter
- Consider reducing `n_samples` for large images or limited GPU memory

### Computation Time
- SmoothGrad/VarGrad are ~n_samples times slower than Grad-CAM
- Parallel processing within each method but sequential across samples
- Consider using smaller `n_samples` for faster iteration

### Colormap Choices
- Grad-CAM/Grad-CAM++: 'jet' colormap (blue to red)
- SmoothGrad: 'hot' colormap (black to yellow/white)  
- VarGrad: 'plasma' colormap (purple to yellow) for distinction

## Dependencies

All new functionality works with existing dependencies:
- PyTorch
- NumPy  
- Matplotlib
- PIL (fallback if OpenCV not available)
- Lightning (PyTorch Lightning)

No additional packages required.

## Backward Compatibility

All existing functionality remains unchanged:
- Original Grad-CAM methods work exactly as before
- Existing scripts and code continue to function
- New methods are additive, not replacing existing ones

## Future Enhancements

Potential improvements:
- Integrated attention mechanisms
- Multi-scale analysis
- Temporal consistency for video data
- Real-time visualization modes
- GPU optimization for batch processing