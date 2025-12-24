#!/bin/bash
# Pre-training validation script
# Run this before submitting training jobs to catch shape errors early

echo "=========================================="
echo "Running Model Validation Tests"
echo "=========================================="

cd "$(dirname "$0")"

# Test model shapes
python pde_examples/test_model_shapes.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All validation tests passed!"
    echo "Ready to submit training job."
    exit 0
else
    echo ""
    echo "❌ Validation tests failed!"
    echo "Fix the errors before training."
    exit 1
fi
