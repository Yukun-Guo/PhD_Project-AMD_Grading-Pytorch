#!/usr/bin/env python3
"""
Memory optimization script for 3D GradCAM processing.
Run this before executing ModelGradCAM_3D.py to optimize CUDA memory usage.
"""

import os
import subprocess
import sys

def setup_cuda_memory_optimization():
    """Setup CUDA memory optimization environment variables."""
    print("Setting up CUDA memory optimization...")
    
    # Set expandable segments to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✓ Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    
    # Additional CUDA memory settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages
    print("✓ Set CUDA_LAUNCH_BLOCKING=1")
    
    # Limit number of threads to reduce memory overhead
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    print("✓ Limited OpenMP and MKL threads to 4")
    
def check_gpu_memory():
    """Check current GPU memory status."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"\n=== GPU Memory Status ===")
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3  # GB
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                free = total_memory - reserved
                
                print(f"GPU {i} ({props.name}):")
                print(f"  Total: {total_memory:.2f} GB")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved: {reserved:.2f} GB")
                print(f"  Free: {free:.2f} GB")
                print(f"  Utilization: {(reserved/total_memory)*100:.1f}%")
        else:
            print("CUDA not available!")
            return False
    except ImportError:
        print("PyTorch not available!")
        return False
    
    return True

def clear_gpu_cache():
    """Clear GPU memory cache aggressively."""
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            # Multiple cleanup passes for better fragmentation handling
            for i in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            
            # Reset memory statistics
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            print("✓ Cleared GPU memory cache aggressively")
        else:
            print("CUDA not available for cache clearing")
    except ImportError:
        print("PyTorch not available for cache clearing")

def run_gradcam_with_optimization(config_path, dataset='all', method='all'):
    """Run ModelGradCAM_3D.py with memory optimization."""
    
    # Setup environment
    setup_cuda_memory_optimization()
    clear_gpu_cache()
    
    # Check initial memory state
    if not check_gpu_memory():
        print("Warning: GPU memory check failed")
    
    # Run the GradCAM script
    cmd = [
        sys.executable, 
        'ModelGradCAM_3D.py',
        '--config', config_path,
        '--dataset', dataset,
        '--method', method
    ]
    
    print(f"\n=== Running Command ===")
    print(" ".join(cmd))
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ ModelGradCAM_3D.py completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ModelGradCAM_3D.py failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-optimized 3D GradCAM runner')
    parser.add_argument('--config', default='configs/config_3d.toml', help='Config file path')
    parser.add_argument('--dataset', default='all', choices=['train', 'val', 'test', 'all'], help='Dataset split')
    parser.add_argument('--method', default='all', choices=['gradcam', 'gradcam++', 'smoothgrad', 'vargrad', 'both', 'all'], help='Visualization method')
    parser.add_argument('--check-only', action='store_true', help='Only check GPU memory status')
    parser.add_argument('--clear-cache', action='store_true', help='Only clear GPU cache')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_gpu_memory()
    elif args.clear_cache:
        clear_gpu_cache()
        check_gpu_memory()
    else:
        success = run_gradcam_with_optimization(args.config, args.dataset, args.method)
        if success:
            print("\n=== Final Memory Status ===")
            check_gpu_memory()
        sys.exit(0 if success else 1)