#!/usr/bin/env python3

"""
Test script to verify the 3D NetModule works correctly
"""

import torch
import toml
from NetModule_3D import NetModule

def test_3d_model():
    """Test the 3D model with sample input"""
    
    # Load config
    config = toml.load('configs/config_3d.toml')
    
    # Create model
    print("Creating 3D NetModule...")
    model = NetModule(config)
    
    # Create sample input [B, C, D, H, W]
    batch_size = 2
    channels = 1  # OCT data is single channel
    depth = 192
    height = 256
    width = 256
    
    sample_input = torch.randn(batch_size, channels, depth, height, width)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [batch_size={batch_size}, n_classes={config['DataModule']['n_class']}]")
    
    # Check output shape
    expected_shape = (batch_size, config['DataModule']['n_class'])
    if output.shape == expected_shape:
        print("✅ Model forward pass successful!")
        print(f"✅ Output shape matches expected: {expected_shape}")
    else:
        print(f"❌ Output shape mismatch. Got {output.shape}, expected {expected_shape}")
        return False
    
    # Test with different batch sizes
    print("\nTesting with different batch sizes...")
    for bs in [1, 3, 4]:
        test_input = torch.randn(bs, channels, depth, height, width)
        with torch.no_grad():
            test_output = model(test_input)
        expected = (bs, config['DataModule']['n_class'])
        if test_output.shape == expected:
            print(f"✅ Batch size {bs}: {test_output.shape}")
        else:
            print(f"❌ Batch size {bs}: got {test_output.shape}, expected {expected}")
            return False
    
    print("\n🎉 All tests passed! The 3D model is working correctly.")
    return True

if __name__ == "__main__":
    test_3d_model()