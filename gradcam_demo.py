"""
Quick Grad-CAM Demo with Synthetic Data

This script demonstrates the Grad-CAM functionality using synthetic data
that matches the structure of your AMD grading model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from NetModule import NetModule
from Utils.grad_cam import GradCAMVisualizer


def create_demo_config():
    """Create a configuration that matches your bio config structure."""
    return {
        'DataModule': {
            'image_shape': [304, 304, 1],  # Same as bio config
            'n_class': 4,
            'k_fold': 5
        },
        'NetModule': {
            'model_name': 'amd_grading_demo',
            'log_dir': './demo_logs/',
            'lr': 0.001,
            'backbone_name': 'efficientnet_b0'  # Smaller for demo
        }
    }


def create_realistic_synthetic_data(batch_size=1):
    """Create synthetic data that resembles medical images."""
    height, width = 304, 304
    
    # Create more realistic looking medical images
    def create_medical_like_image():
        # Base noise
        img = torch.randn(batch_size, 1, height, width) * 0.3
        
        # Add some circular/oval structures (like vessels or lesions)
        y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
        
        # Add a few circular features at different locations
        for i in range(3):
            center_y = torch.rand(1) * 0.6 - 0.3
            center_x = torch.rand(1) * 0.6 - 0.3
            radius = torch.rand(1) * 0.2 + 0.1
            intensity = torch.rand(1) * 0.5 + 0.5
            
            circular_feature = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * radius**2))
            img += circular_feature.unsqueeze(0).unsqueeze(0) * intensity
        
        # Normalize to [0, 1] range
        img = torch.clamp(img, 0, 1)
        return img
    
    # Generate different types of images
    mnv = create_medical_like_image()      # MNV (microvascular networks)
    fluid = create_medical_like_image()    # Fluid accumulation
    ga = create_medical_like_image()       # Geographic atrophy
    drusen = create_medical_like_image()   # Drusen deposits
    
    return mnv, fluid, ga, drusen


def demo_single_analysis():
    """Demonstrate single sample analysis."""
    print("🔬 GRAD-CAM DEMO: Single Sample Analysis")
    print("="*50)
    
    # Create model
    config = create_demo_config()
    model = NetModule(config)
    model.eval()
    
    # Generate realistic synthetic data
    mnv, fluid, ga, drusen = create_realistic_synthetic_data()
    
    print(f"Generated synthetic medical images:")
    print(f"  MNV: {mnv.shape} (microvascular networks)")
    print(f"  Fluid: {fluid.shape} (fluid accumulation)")
    print(f"  GA: {ga.shape} (geographic atrophy)")  
    print(f"  Drusen: {drusen.shape} (drusen deposits)")
    
    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nAnalyzing with Grad-CAM...")
        
        # Perform analysis
        results = model.analyze_prediction_with_gradcam(
            mnv, fluid, ga, drusen,
            target_class=None,  # Use predicted class
            save_dir=temp_dir,
            sample_id="demo_patient"
        )
        
        # Show results
        print(f"\n📊 Analysis Results:")
        print(f"  Predicted AMD Grade: {results['predicted_class']}")
        print(f"  Confidence: {results['confidence']:.1%}")
        print(f"  Analysis Target: Class {results['target_class']}")
        
        # Show class probabilities
        probs = results['probabilities'][0]
        class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
        print(f"\n📈 Class Probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
            print(f"  {class_name}: {prob:.1%}")
        
        # Show heatmap info
        print(f"\n🔥 Generated Heatmaps:")
        for input_name, heatmap in results['heatmaps'].items():
            activation_strength = heatmap.max()
            print(f"  {input_name.upper()}: Max activation = {activation_strength:.3f}")
        
        # Copy files to a persistent location for viewing
        demo_dir = Path("./gradcam_demo_results")
        demo_dir.mkdir(exist_ok=True)
        
        import shutil
        for file in Path(temp_dir).glob("*"):
            if file.is_file():
                shutil.copy2(file, demo_dir / file.name)
            elif file.is_dir():
                shutil.copytree(file, demo_dir / file.name, dirs_exist_ok=True)
        
        print(f"\n💾 Results saved to: {demo_dir.absolute()}")
        print(f"   - Main visualization: demo_patient_gradcam_visualization.png")
        print(f"   - Individual heatmaps: demo_patient_heatmaps/")


def demo_interpretation_guide():
    """Print interpretation guide for the results."""
    print("\n" + "="*60)
    print("🧠 HOW TO INTERPRET GRAD-CAM RESULTS")
    print("="*60)
    
    print("""
📍 HEATMAP COLORS:
  🔴 Red regions    → High importance for classification
  🟡 Yellow regions → Moderate importance  
  🔵 Blue regions   → Low importance
  ⚪ White regions  → Neutral/background

🩺 MEDICAL INTERPRETATION:
  • MNV heatmap    → Shows areas affecting microvascular networks
  • Fluid heatmap  → Highlights regions with fluid accumulation
  • GA heatmap     → Points to geographic atrophy patterns
  • Drusen heatmap → Identifies drusen deposits importance

⚖️ CLINICAL SIGNIFICANCE:
  • High activation in pathological regions suggests model is learning
    relevant anatomical features for AMD grading
  • Scattered activation may indicate early-stage changes
  • Focused activation often corresponds to severe pathology

🎯 WHAT TO LOOK FOR:
  ✅ Activation in clinically relevant areas (fovea, vessels, lesions)
  ✅ Different patterns for different AMD grades
  ✅ Consistency with clinical knowledge
  
  ⚠️  Random/scattered activation → May need more training
  ⚠️  No activation → Check model or preprocessing
  ⚠️  Too much activation → May be overfitting
""")


def main():
    """Run the complete demo."""
    print("🚀 STARTING GRAD-CAM DEMONSTRATION")
    print("=" * 60)
    print("This demo will:")
    print("1. Create a model matching your AMD grading architecture")
    print("2. Generate synthetic medical-like images")  
    print("3. Perform Grad-CAM analysis")
    print("4. Show interpretation of results")
    print("5. Save visualizations for inspection")
    
    try:
        # Run the demo
        demo_single_analysis()
        demo_interpretation_guide()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"""
🎉 Next Steps:
1. Check the generated files in ./gradcam_demo_results/
2. Run with your real data:
   python GradCAM_Example.py --config configs/config_bio.toml
3. Integrate into your workflow:
   results = model.analyze_prediction_with_gradcam(mnv, fluid, ga, drusen)

📚 For detailed documentation, see: docs/GradCAM_Guide.md
""")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()