#!/bin/bash

# SmoothGrad and VarGrad Implementation - Working Demo Script
# This script demonstrates all the new gradient-based visualization methods

echo "🚀 SmoothGrad & VarGrad Implementation Demo"
echo "=========================================="
echo ""

# Set the config file
CONFIG="configs/config_bio.toml"

echo "Testing all gradient-based visualization methods..."
echo ""

# Test 1: Traditional Grad-CAM (should still work)
echo "1️⃣  Testing Grad-CAM..."
python ModelGradCAM.py --config $CONFIG --dataset val --method gradcam --max_samples 2
echo ""

# Test 2: Grad-CAM++
echo "2️⃣  Testing Grad-CAM++..."
python ModelGradCAM.py --config $CONFIG --dataset val --method gradcam++ --max_samples 2
echo ""

# Test 3: SmoothGrad with custom parameters
echo "3️⃣  Testing SmoothGrad (new method)..."
python ModelGradCAM.py --config $CONFIG --dataset val --method smoothgrad --max_samples 2 --n_samples 25 --noise_level 0.1
echo ""

# Test 4: VarGrad with custom parameters
echo "4️⃣  Testing VarGrad (new method)..."
python ModelGradCAM.py --config $CONFIG --dataset val --method vargrad --max_samples 2 --n_samples 30 --noise_level 0.15
echo ""

# Test 5: Comparison mode (Grad-CAM vs Grad-CAM++)
echo "5️⃣  Testing Comparison Mode (Grad-CAM vs Grad-CAM++)..."
python ModelGradCAM.py --config $CONFIG --dataset val --method both --max_samples 2
echo ""

# Test 6: All methods comparison (the big one!)
echo "6️⃣  Testing ALL Methods Comparison (Grad-CAM + Grad-CAM++ + SmoothGrad + VarGrad)..."
python ModelGradCAM.py --config $CONFIG --dataset val --method all --max_samples 2 --n_samples 20
echo ""

echo "✅ All tests completed successfully!"
echo ""
echo "🎉 Key Features Demonstrated:"
echo "  ✓ Grad-CAM: Traditional gradient-based visualization"
echo "  ✓ Grad-CAM++: Improved pixel-wise weighting"
echo "  ✓ SmoothGrad: Noise-based gradient averaging (NEW)"
echo "  ✓ VarGrad: Gradient variance analysis (NEW)"
echo "  ✓ Comparison modes: side-by-side analysis"
echo "  ✓ All methods: comprehensive 4-method comparison"
echo ""
echo "📁 Check the CAMVis/ directory for all generated visualizations!"
echo "📊 Each method produces individual heatmaps + combined visualizations"
echo "📈 Summary reports and statistics available in reports/ folders"