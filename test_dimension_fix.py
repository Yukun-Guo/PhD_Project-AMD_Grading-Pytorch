#!/usr/bin/env python3
"""
Test script to verify the dimension correction for 3D OCT volumes.
This script tests the dimension transposition logic without running the full model.
"""

import numpy as np

def test_dimension_correction():
    """Test the dimension correction logic"""
    print("Testing dimension correction logic...")
    
    # Test case 1: Volume with shape (H, W, D) = (224, 224, 256) that needs transpose to (D, H, W)
    print("\n=== Test Case 1: (H, W, D) -> (D, H, W) ===")
    test_volume_hwl = np.random.rand(224, 224, 256)
    print(f"Original shape: {test_volume_hwl.shape}")
    
    # Apply the correction logic
    if test_volume_hwl.shape[0] != 256 and test_volume_hwl.shape[2] == 256:
        corrected_volume = np.transpose(test_volume_hwl, (2, 0, 1))  # (H, W, D) -> (D, H, W)
        print(f"Transposed to correct shape: {corrected_volume.shape}")
        assert corrected_volume.shape[0] == 256, f"Expected 256 slices, got {corrected_volume.shape[0]}"
        print("✓ Transpose applied correctly")
    else:
        print("✗ Transpose logic did not trigger")
    
    # Test case 2: Volume already in correct shape (D, H, W) = (256, 224, 224)
    print("\n=== Test Case 2: Already correct (D, H, W) ===")
    test_volume_dhw = np.random.rand(256, 224, 224)
    print(f"Original shape: {test_volume_dhw.shape}")
    
    # Apply the correction logic
    if test_volume_dhw.shape[0] != 256 and test_volume_dhw.shape[2] == 256:
        corrected_volume = np.transpose(test_volume_dhw, (2, 0, 1))
        print(f"Transposed to: {corrected_volume.shape}")
    else:
        print("✓ No transpose needed - already correct shape")
    
    # Test case 3: Different dimensions that shouldn't be transposed
    print("\n=== Test Case 3: Different dimensions ===")
    test_volume_other = np.random.rand(192, 224, 224)
    print(f"Original shape: {test_volume_other.shape}")
    
    # Apply the correction logic
    if test_volume_other.shape[0] != 256 and test_volume_other.shape[2] == 256:
        corrected_volume = np.transpose(test_volume_other, (2, 0, 1))
        print(f"Transposed to: {corrected_volume.shape}")
    else:
        print("✓ No transpose applied - doesn't match 256-slice pattern")
    
    print("\n=== Summary ===")
    print("Dimension correction logic:")
    print("- If shape[0] != 256 AND shape[2] == 256:")
    print("  → Apply transpose(2, 0, 1) to move depth dimension to front")
    print("- Otherwise: Keep original shape")
    print("- Result should always have 256 slices in first dimension for OCT volumes")

if __name__ == "__main__":
    test_dimension_correction()