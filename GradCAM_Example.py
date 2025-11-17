"""
Example script demonstrating Grad-CAM visualization for AMD grading model.

This script shows how to:
1. Load a trained model
2. Load sample data
3. Generate Grad-CAM visualizations for each input (MNV, Fluid, GA, Drusen)
4. Save and display the results

Usage:
    python GradCAM_Example.py --config configs/config_bio.toml --sample_idx 0
"""

import argparse
import torch
import toml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import project modules
from NetModule import NetModule
from DataModule import DataModel
from Utils.grad_cam import GradCAMVisualizer


def find_latest_checkpoint(checkpoint_dir: str, model_name: str) -> str:
    """Find the latest checkpoint file."""
    checkpoint_path = Path(checkpoint_dir) / model_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Look for .ckpt files
    ckpt_files = list(checkpoint_path.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_path}")
    
    # Return the most recent checkpoint
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    return str(latest_ckpt)


def load_model_and_data(config_path: str):
    """Load the trained model and validation data."""
    print(f"Loading configuration from: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Setup data module
    print("Setting up data module...")
    data_model = DataModel(config=config)
    data_model.setup()
    
    # Find and load model checkpoint
    print("Loading model checkpoint...")
    checkpoint_path = find_latest_checkpoint(
        config['NetModule']['log_dir'], 
        config['NetModule']['model_name']
    )
    
    model = NetModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    
    return model, data_model, config


def analyze_single_sample(model, data_loader, sample_idx: int = 0, save_dir: str = "./grad_cam_results"):
    """Analyze a single sample with Grad-CAM."""
    
    # Get sample
    print(f"Extracting sample {sample_idx} from data loader...")
    samples_seen = 0
    target_batch = None
    
    for batch in data_loader:
        batch_size = batch[0].size(0)  # mnv tensor batch size
        if samples_seen + batch_size > sample_idx:
            # Found the batch containing our sample
            target_batch = batch
            sample_in_batch = sample_idx - samples_seen
            break
        samples_seen += batch_size
    
    if target_batch is None:
        raise IndexError(f"Sample index {sample_idx} not found in data loader")
    
    # Extract the specific sample
    mnv, fluid, ga, drusen, y, sample_ids = target_batch
    mnv = mnv[sample_in_batch:sample_in_batch+1]
    fluid = fluid[sample_in_batch:sample_in_batch+1]
    ga = ga[sample_in_batch:sample_in_batch+1]
    drusen = drusen[sample_in_batch:sample_in_batch+1]
    true_class = int(y[sample_in_batch].item())
    sample_id = sample_ids[sample_in_batch] if isinstance(sample_ids, list) else f"sample_{sample_idx}"
    
    print(f"Sample ID: {sample_id}")
    print(f"True class: {true_class}")
    print(f"Input shapes: MNV: {mnv.shape}, Fluid: {fluid.shape}, GA: {ga.shape}, Drusen: {drusen.shape}")
    
    # Analyze with Grad-CAM
    print("Generating Grad-CAM analysis...")
    analysis_results = model.analyze_prediction_with_gradcam(
        mnv, fluid, ga, drusen,
        target_class=None,  # Use predicted class
        save_dir=save_dir,
        sample_id=sample_id
    )
    
    # Print results
    print("\n" + "="*50)
    print("GRAD-CAM ANALYSIS RESULTS")
    print("="*50)
    print(f"Sample ID: {sample_id}")
    print(f"True Class: {true_class}")
    print(f"Predicted Class: {analysis_results['predicted_class']}")
    print(f"Confidence: {analysis_results['confidence']:.4f}")
    print(f"Target Class for Analysis: {analysis_results['target_class']}")
    
    # Print class probabilities
    probs = analysis_results['probabilities'][0]
    print(f"\nClass Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  Class {i}: {prob:.4f}")
    
    # Print heatmap information
    print(f"\nGenerated Heatmaps:")
    for input_name in ['mnv', 'fluid', 'ga', 'drusen']:
        if input_name in analysis_results['heatmaps']:
            heatmap = analysis_results['heatmaps'][input_name]
            print(f"  {input_name.upper()}: shape {heatmap.shape}, range [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        else:
            print(f"  {input_name.upper()}: Not available")
    
    return analysis_results


def batch_analysis(model, data_loader, num_samples: int = 5, save_dir: str = "./grad_cam_batch_results"):
    """Perform batch analysis on multiple samples."""
    
    print(f"Performing batch analysis on {num_samples} samples...")
    
    # Create visualizer
    visualizer = GradCAMVisualizer(model)
    
    # Run batch analysis
    results = visualizer.batch_analyze(
        data_loader, 
        num_samples=num_samples, 
        save_dir=Path(save_dir)
    )
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH ANALYSIS SUMMARY")
    print("="*50)
    
    correct_predictions = sum(1 for r in results if r['true_class'] == r['pred_class'])
    accuracy = correct_predictions / len(results)
    
    print(f"Total samples analyzed: {len(results)}")
    print(f"Correct predictions: {correct_predictions}/{len(results)} ({accuracy:.2%})")
    
    # Class distribution
    true_classes = [r['true_class'] for r in results]
    pred_classes = [r['pred_class'] for r in results]
    
    print(f"\nTrue class distribution: {dict(zip(*np.unique(true_classes, return_counts=True)))}")
    print(f"Predicted class distribution: {dict(zip(*np.unique(pred_classes, return_counts=True)))}")
    
    # Cleanup
    visualizer.cleanup()
    
    return results


def create_summary_visualization(results, save_path: str = "./grad_cam_summary.png"):
    """Create a summary visualization of multiple samples."""
    
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    input_names = ['mnv', 'fluid', 'ga', 'drusen']
    display_names = ['MNV', 'Fluid', 'GA', 'Drusen']
    
    for i, result in enumerate(results):
        sample_id = result['sample_id']
        true_class = result['true_class']
        pred_class = result['pred_class']
        heatmaps = result['heatmaps']
        
        for j, (inp_name, display_name) in enumerate(zip(input_names, display_names)):
            if inp_name in heatmaps:
                heatmap = heatmaps[inp_name]
                im = axes[i, j].imshow(heatmap, cmap='jet', aspect='auto')
                axes[i, j].set_title(f'{display_name}\n{sample_id}')
            else:
                axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', 
                               transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{display_name}\n{sample_id}')
            
            axes[i, j].axis('off')
        
        # Add sample information on the left
        fig.text(0.02, 1 - (i + 0.5) / n_samples, 
                f'True: {true_class}\nPred: {pred_class}', 
                va='center', ha='left', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.suptitle(f'Grad-CAM Summary - {n_samples} Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Analysis for AMD Grading Model')
    parser.add_argument('--config', type=str, default='configs/config_bio.toml',
                      help='Path to configuration file')
    parser.add_argument('--sample_idx', type=int, default=0,
                      help='Index of sample to analyze (for single sample analysis)')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                      help='Analysis mode: single sample or batch analysis')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='Number of samples for batch analysis')
    parser.add_argument('--save_dir', type=str, default='./grad_cam_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    try:
        # Load model and data
        model, data_model, config = load_model_and_data(args.config)
        
        # Get validation data loader
        val_loader = data_model.val_dataloader()
        
        # Create save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"{args.save_dir}_{timestamp}"
        
        if args.mode == 'single':
            # Single sample analysis
            results = analyze_single_sample(
                model, val_loader, 
                sample_idx=args.sample_idx,
                save_dir=save_dir
            )
            
            print(f"\nResults saved to: {save_dir}")
            
        elif args.mode == 'batch':
            # Batch analysis
            results = batch_analysis(
                model, val_loader,
                num_samples=args.num_samples,
                save_dir=save_dir
            )
            
            # Create summary visualization
            summary_path = f"{save_dir}/summary_visualization.png"
            create_summary_visualization(results, summary_path)
            
            print(f"\nBatch results saved to: {save_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()