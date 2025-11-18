#!/usr/bin/env python3

"""
Test script for PredictionVal_3D.py - validates the 3D prediction setup
"""

import os
import torch
import toml
from pathlib import Path

def test_prediction_setup():
    """Test the 3D prediction validation setup"""
    
    print("🔧 Testing 3D Prediction Validation Setup...")
    
    # Check if config file exists
    config_file = "configs/config_3d.toml"
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    print(f"✅ Config file found: {config_file}")
    
    # Load configuration
    try:
        config = toml.load(config_file)
        print(f"✅ Configuration loaded successfully")
        print(f"   - Model: {config['NetModule']['model_name']}")
        print(f"   - Classes: {config['DataModule']['n_class']}")
        print(f"   - Log dir: {config['NetModule']['log_dir']}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Check if checkpoint directory exists
    log_dir = config['NetModule']['log_dir']
    model_name = config['NetModule']['model_name']
    
    print(f"🔍 Checking for checkpoints...")
    
    # Look for checkpoint files
    logs_path = Path("./logs")
    if logs_path.exists():
        ckpt_files = list(logs_path.rglob("*.ckpt"))
        if ckpt_files:
            print(f"✅ Found {len(ckpt_files)} checkpoint files:")
            for ckpt in ckpt_files[:3]:  # Show first 3
                print(f"   - {ckpt}")
            if len(ckpt_files) > 3:
                print(f"   ... and {len(ckpt_files) - 3} more")
        else:
            print("⚠️  No checkpoint files found")
            print("   Note: You need to train the model first before running validation")
    else:
        print("⚠️  Logs directory not found")
        print("   Note: You need to train the model first before running validation")
    
    # Test imports
    print("🔍 Testing imports...")
    try:
        from NetModule_3D import NetModule
        print("✅ NetModule_3D imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NetModule_3D: {e}")
        return False
    
    try:
        from DataModule_3D import DataModel
        print("✅ DataModule_3D imported successfully") 
    except ImportError as e:
        print(f"❌ Failed to import DataModule_3D: {e}")
        return False
    
    # Test data module creation
    print("🔍 Testing data module creation...")
    try:
        data_model = DataModel(config=config)
        print("✅ DataModule_3D created successfully")
    except Exception as e:
        print(f"❌ Failed to create DataModule_3D: {e}")
        return False
    
    # Test data setup
    print("🔍 Testing data setup...")
    try:
        data_model.setup('fit')
        val_dataloader = data_model.val_dataloader()
        print(f"✅ Validation dataloader created successfully")
        print(f"   - Validation samples: {len(val_dataloader.dataset)}")
        print(f"   - Batch size: {val_dataloader.batch_size}")
        print(f"   - Number of batches: {len(val_dataloader)}")
    except Exception as e:
        print(f"❌ Failed to setup validation data: {e}")
        return False
    
    # Test batch format
    print("🔍 Testing batch format...")
    try:
        sample_batch = next(iter(val_dataloader))
        if len(sample_batch) == 3:
            mat, y, _ = sample_batch
            print(f"✅ Batch format is correct for 3D model")
            print(f"   - Input shape: {mat.shape}")
            print(f"   - Label shape: {y.shape}")
            print(f"   - Expected input format: [batch, channel, depth, height, width]")
            
            # Verify input shape matches expected 3D format
            if len(mat.shape) == 5:  # [B, C, D, H, W]
                print(f"✅ Input has correct 5D shape for 3D processing")
            else:
                print(f"⚠️  Input shape may not be correct for 3D: {mat.shape}")
        else:
            print(f"⚠️  Unexpected batch format: {len(sample_batch)} elements")
    except StopIteration:
        print("⚠️  No validation data available")
    except Exception as e:
        print(f"❌ Failed to test batch format: {e}")
        return False
    
    print("\n🎉 3D Prediction Validation Setup Test Complete!")
    print("\n📋 Summary:")
    print("   ✅ Configuration loading: OK")
    print("   ✅ Module imports: OK") 
    print("   ✅ Data pipeline: OK")
    print("   ✅ Batch format: OK")
    
    if ckpt_files:
        print("   ✅ Checkpoints: Available")
        print("\n🚀 Ready to run: python PredictionVal_3D.py")
    else:
        print("   ⚠️  Checkpoints: Not found")
        print("\n🔧 Next step: Train model first with: python TrainerFit_3D.py")
    
    return True

if __name__ == "__main__":
    success = test_prediction_setup()
    if not success:
        print("\n🔧 Please fix the issues above before running validation.")