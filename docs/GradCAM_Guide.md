# Grad-CAM Implementation Guide

## Overview

This implementation provides comprehensive Grad-CAM (Gradient-weighted Class Activation Mapping) visualization for your multi-input EfficientNet-based AMD grading model. The implementation generates heat maps showing which regions of each input image (MNV, Fluid, GA, Drusen) are most important for the model's classification decisions.

## Key Features

- **Multi-Input Support**: Generates separate heat maps for each of the 4 input branches
- **EfficientNet Compatibility**: Works with all EfficientNet variants (B0-B7)
- **Easy Integration**: Seamlessly integrates with existing PyTorch Lightning model
- **Flexible Visualization**: Multiple output formats and customization options
- **Batch Processing**: Analyze multiple samples efficiently

## Architecture

The Grad-CAM implementation consists of three main components:

### 1. `MultiInputGradCAM` Class
Core implementation that handles:
- Hook registration for gradient and activation capture
- CAM computation for each input branch
- Heat map generation and normalization

### 2. `GradCAMVisualizer` Class
High-level interface providing:
- Simplified analysis workflow
- Batch processing capabilities
- Automatic visualization and saving

### 3. `NetModule` Integration
Direct methods added to your existing model:
- `create_grad_cam_visualizer()`: Create visualizer instance
- `analyze_prediction_with_gradcam()`: One-line analysis method

## Usage Examples

### Quick Start

```python
import torch
from NetModule import NetModule

# Load your trained model
model = NetModule.load_from_checkpoint('path/to/checkpoint.ckpt', config=config)

# Prepare your inputs (batch size = 1)
mnv = torch.randn(1, 1, 224, 224)      # MNV image
fluid = torch.randn(1, 1, 224, 224)    # Fluid image  
ga = torch.randn(1, 1, 224, 224)       # GA image
drusen = torch.randn(1, 1, 224, 224)   # Drusen image

# Generate Grad-CAM analysis
results = model.analyze_prediction_with_gradcam(
    mnv, fluid, ga, drusen,
    save_dir="./gradcam_results",
    sample_id="patient_001"
)

print(f"Predicted class: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.4f}")
print(f"Available heatmaps: {list(results['heatmaps'].keys())}")
```

### Advanced Usage

```python
from Utils.grad_cam import GradCAMVisualizer

# Create visualizer with custom settings
visualizer = GradCAMVisualizer(model, use_cuda=True)

# Analyze specific target class
heatmaps = visualizer.analyze_sample(
    inputs=[mnv, fluid, ga, drusen],
    target_class=2,  # Focus on class 2
    save_dir="./class2_analysis",
    sample_id="analysis_001"
)

# Access individual heatmaps
mnv_heatmap = heatmaps['mnv']          # Shape: [H, W]
fluid_heatmap = heatmaps['fluid']      # Shape: [H, W]
ga_heatmap = heatmaps['ga']            # Shape: [H, W]  
drusen_heatmap = heatmaps['drusen']    # Shape: [H, W]

# Cleanup when done
visualizer.cleanup()
```

### Batch Processing

```python
# Analyze multiple samples from data loader
results = visualizer.batch_analyze(
    dataloader=val_dataloader,
    num_samples=10,
    save_dir="./batch_analysis"
)

# Results contain analysis for each sample
for result in results:
    print(f"Sample: {result['sample_id']}")
    print(f"True: {result['true_class']}, Pred: {result['pred_class']}")
    print(f"Heatmaps: {list(result['heatmaps'].keys())}")
```

## Command Line Usage

### Single Sample Analysis
```bash
python GradCAM_Example.py --config configs/config_bio.toml --sample_idx 5
```

### Batch Analysis  
```bash
python GradCAM_Example.py --mode batch --num_samples 10 --config configs/config_bio.toml
```

### Custom Save Directory
```bash
python GradCAM_Example.py --save_dir ./my_gradcam_results --sample_idx 0
```

## Output Files Structure

When you run Grad-CAM analysis, the following files are created:

```
gradcam_results_20241113_143022/
├── sample_001_gradcam_visualization.png    # Combined visualization
├── sample_001_heatmaps/                    # Individual heatmaps folder
│   ├── sample_001_mnv.png                  # MNV heatmap
│   ├── sample_001_fluid.png                # Fluid heatmap  
│   ├── sample_001_ga.png                   # GA heatmap
│   └── sample_001_drusen.png               # Drusen heatmap
└── summary_visualization.png               # Batch summary (if batch mode)
```

## Understanding the Visualizations

### Heat Map Interpretation

- **Red regions**: High importance for classification decision
- **Blue regions**: Low importance for classification decision  
- **Yellow/Green**: Moderate importance

### Visualization Layout

The main visualization shows:
- **Top row**: Original input images (MNV, Fluid, GA, Drusen)
- **Bottom row**: Grad-CAM overlays on original images

### Analysis Metrics

Each analysis provides:
- `predicted_class`: Model's predicted class (0-3)
- `confidence`: Prediction confidence (0-1)
- `probabilities`: Full probability distribution across classes
- `heatmaps`: Dictionary of heat maps for each input

## Technical Details

### Target Layers

The implementation targets the feature extraction layers of each EfficientNet backbone:
- `backbone1`: For MNV input processing
- `backbone2`: For Fluid input processing  
- `backbone3`: For GA input processing
- `backbone4`: For Drusen input processing

### Heat Map Generation Process

1. **Forward Pass**: Input images through model to get predictions
2. **Backward Pass**: Compute gradients w.r.t. target class  
3. **Gradient Capture**: Extract gradients from target layers
4. **Activation Capture**: Extract feature activations from target layers
5. **Weight Computation**: Global average pooling of gradients
6. **CAM Generation**: Weighted sum of activations using gradient weights
7. **Normalization**: Apply ReLU and normalize to [0,1] range
8. **Resizing**: Resize heat map to match original input size

### Memory Considerations

- Each analysis requires gradient computation (higher memory usage)
- Use batch size = 1 for individual sample analysis
- Call `cleanup()` or `remove_hooks()` to free memory when done
- Consider using CPU mode for large batch processing

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Use CPU mode
visualizer = GradCAMVisualizer(model, use_cuda=False)
```

**2. Layer Not Found Warning**
```
Warning: Layer 'backbone1' not found in model
```
Check that your model architecture matches the expected structure.

**3. Empty Heat Maps**
- Ensure model is in evaluation mode: `model.eval()`
- Check that gradients are enabled during analysis
- Verify target class is valid (0 to n_classes-1)

### Debugging Tips

**Enable detailed output:**
```python
# Check model structure
print(model)

# Verify input shapes
print(f"Input shapes: {[inp.shape for inp in inputs]}")

# Check layer availability  
for layer_name in ['backbone1', 'backbone2', 'backbone3', 'backbone4']:
    layer = getattr(model.out, layer_name, None)
    print(f"{layer_name}: {'Found' if layer else 'Not found'}")
```

**Test with synthetic data:**
```bash
python test_gradcam.py
```

## Performance Optimization

### For Single Sample Analysis
- Use `model.analyze_prediction_with_gradcam()` for simplicity
- Set `use_cuda=True` if GPU memory allows
- Process one sample at a time for memory efficiency

### For Batch Analysis  
- Use `GradCAMVisualizer.batch_analyze()` for efficiency
- Consider `use_cuda=False` for large batches
- Save intermediate results to avoid recomputation

### Memory Management
```python
# Always cleanup when done
try:
    results = visualizer.analyze_sample(inputs)
    # Process results...
finally:
    visualizer.cleanup()  # Important for memory management
```

## Integration with Existing Workflow

### With Training Pipeline
```python
# After training completion
best_model_path = trainer.checkpoint_callback.best_model_path
model = NetModule.load_from_checkpoint(best_model_path, config=config)

# Analyze validation samples
val_loader = data_module.val_dataloader()
visualizer = GradCAMVisualizer(model)
results = visualizer.batch_analyze(val_loader, num_samples=20)
```

### With Inference Pipeline  
```python
# During inference
prediction_results = model.predict(inputs)
gradcam_results = model.analyze_prediction_with_gradcam(
    *inputs, 
    target_class=prediction_results['predicted_class']
)

# Combine for comprehensive analysis
final_results = {**prediction_results, **gradcam_results}
```

## API Reference

### MultiInputGradCAM

```python
class MultiInputGradCAM:
    def __init__(self, model, target_layers: List[str], use_cuda: bool = True)
    def generate_cam(self, inputs: List[torch.Tensor], target_class: Optional[int] = None) -> Dict[str, np.ndarray]
    def visualize_cam(self, inputs: List[torch.Tensor], heatmaps: Dict[str, np.ndarray], save_path: Optional[Path] = None)
    def save_heatmaps(self, heatmaps: Dict[str, np.ndarray], save_dir: Path, filename_prefix: str = "heatmap")
    def remove_hooks(self)
```

### GradCAMVisualizer

```python
class GradCAMVisualizer:
    def __init__(self, model, use_cuda: bool = True)
    def analyze_sample(self, inputs: List[torch.Tensor], target_class: Optional[int] = None, save_dir: Optional[Path] = None, sample_id: str = "sample") -> Dict[str, np.ndarray]
    def batch_analyze(self, dataloader, num_samples: int = 10, save_dir: Optional[Path] = None) -> List[Dict]
    def cleanup(self)
```

### NetModule Extensions

```python
class NetModule(L.LightningModule):
    def create_grad_cam_visualizer(self, use_cuda: bool = True) -> GradCAMVisualizer
    def analyze_prediction_with_gradcam(self, mnv: torch.Tensor, fluid: torch.Tensor, ga: torch.Tensor, drusen: torch.Tensor, target_class: Optional[int] = None, save_dir: Optional[str] = None, sample_id: str = "sample") -> Dict[str, Any]
```

## Next Steps

1. **Test the Implementation**: Run `python test_gradcam.py` to verify everything works
2. **Try Examples**: Use `python GradCAM_Example.py` with your data
3. **Integrate**: Add Grad-CAM analysis to your existing validation/inference pipeline
4. **Customize**: Modify visualization styles or add additional analysis metrics

For questions or issues, refer to the troubleshooting section or check the test implementation for usage patterns.