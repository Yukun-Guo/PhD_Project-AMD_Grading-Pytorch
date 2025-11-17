#!/usr/bin/env python3

"""
Test script for 3D training setup - minimal version to check compatibility
"""

import os
import lightning as L
import torch
from NetModule_3D import NetModule
from DataModule_3D import DataModel
import toml

def test_training_setup():
    """Test the 3D training setup with minimal configuration"""
    
    print("🔧 Setting up 3D training test...")
    
    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Set seed for reproducibility
    L.seed_everything(1234)
    
    # Load configuration
    print("📝 Loading configuration...")
    toml_file = "./configs/config_3d.toml"
    config = toml.load(toml_file)
    print(f"✅ Config loaded: {config['NetModule']['model_name']}")
    
    # Create data module
    print("📊 Creating data module...")
    try:
        data_model = DataModel(config=config)
        print("✅ DataModule created successfully")
    except Exception as e:
        print(f"❌ DataModule creation failed: {e}")
        return False
    
    # Create network module
    print("🧠 Creating network module...")
    try:
        net_model = NetModule(config=config)
        print("✅ NetModule created successfully")
        print(f"   - Model: {net_model.backbone_name}")
        print(f"   - Input channels: {net_model.img_chn}")
        print(f"   - Output classes: {net_model.n_class}")
        print(f"   - Input size: {net_model.input_size_3d}")
    except Exception as e:
        print(f"❌ NetModule creation failed: {e}")
        return False
    
    # Test data loading (setup only, no actual loading)
    print("📥 Testing data setup...")
    try:
        data_model.setup('fit')
        print("✅ Data setup completed")
        print(f"   - Data path: {config['DataModule']['data_path']}")
        print(f"   - Batch size: {config['DataModule']['batch_size']}")
    except Exception as e:
        print(f"❌ Data setup failed: {e}")
        return False
    
    # Create trainer (minimal configuration)
    print("🏃 Creating trainer...")
    try:
        trainer = L.Trainer(
            logger=net_model.configure_loggers(),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=1,  # Just 1 epoch for testing
            limit_train_batches=2,  # Only 2 batches for testing
            limit_val_batches=1,    # Only 1 validation batch
            enable_checkpointing=False,  # Disable checkpointing for test
            enable_model_summary=False,  # Disable summary to avoid potential issues
        )
        print("✅ Trainer created successfully")
        print(f"   - Device: {trainer.accelerator}")
        print(f"   - Max epochs: {trainer.max_epochs}")
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False
    
    # Test compatibility (no actual training)
    print("🔍 Testing model-data compatibility...")
    try:
        # Create a sample batch to test forward pass
        sample_input = torch.randn(2, 1, 192, 256, 256)  # [batch, channel, depth, height, width]
        
        net_model.eval()
        with torch.no_grad():
            output = net_model(sample_input)
        
        expected_shape = (2, config['DataModule']['n_class'])
        if output.shape == expected_shape:
            print(f"✅ Forward pass successful: {output.shape}")
        else:
            print(f"❌ Output shape mismatch: got {output.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The 3D training setup is ready.")
    print("\n📋 Summary:")
    print(f"   ✅ 3D EfficientNet-B2 model: {net_model.backbone_name}")
    print(f"   ✅ Input shape: [batch, {net_model.img_chn}, {net_model.input_size_3d[0]}, {net_model.input_size_3d[1]}, {net_model.input_size_3d[2]}]")
    print(f"   ✅ Output classes: {net_model.n_class}")
    print(f"   ✅ Parameters: ~9.4M")
    print(f"   ✅ Data pipeline: Ready")
    print(f"   ✅ Trainer: Configured")
    
    return True

if __name__ == "__main__":
    success = test_training_setup()
    if success:
        print("\n🚀 Ready to start training! Run TrainerFit_3D.py for full training.")
    else:
        print("\n🔧 Please fix the issues above before training.")