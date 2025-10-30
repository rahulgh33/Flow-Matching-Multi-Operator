"""
Quick inpainting evaluation script - just generates results without training.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from conditional.fm_cifar_inpainting import (
    InpaintingUNet, create_random_mask, inpaint_image, 
    visualize_inpainting_results, create_cifar_inpainting_loader
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model (same as training)
    model = InpaintingUNet(channels=3, base_channels=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load some test data
    train_loader = create_cifar_inpainting_loader(batch_size=32)
    test_data = next(iter(train_loader))[0][:8].to(device)
    
    # Create test masks
    test_masks = create_random_mask(8, 32, 32, hole_size_range=(10, 20), num_holes_range=(1, 2)).to(device)
    test_masked = test_data * test_masks
    
    print("Generating inpainting results with random weights (for testing)...")
    
    # Inpaint with current (random) weights - just to test the pipeline
    inpainted_results = []
    for i in range(8):
        inpainted = inpaint_image(
            model, 
            test_masked[i:i+1], 
            test_masks[i:i+1], 
            device, 
            num_steps=15
        )
        inpainted_results.append(inpainted)
    
    inpainted_batch = torch.cat(inpainted_results, dim=0)
    
    # Create results directory
    os.makedirs("results/cifar", exist_ok=True)
    
    # Visualize results
    visualize_inpainting_results(
        test_data, test_masked, test_masks, inpainted_batch,
        save_path="results/cifar/inpainting_test.png"
    )
    
    print("Test completed! Check results/cifar/inpainting_test.png")
    print("Note: This uses random weights, so results won't be good.")
    print("For real results, you need the trained model weights.")

if __name__ == "__main__":
    main()