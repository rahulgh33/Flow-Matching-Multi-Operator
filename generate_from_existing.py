"""
Generate samples from existing trained model (works with old save format).
"""

import torch
import sys
import os

# Add conditional directory to path
sys.path.append('conditional')
from fm_cifar_conditional import UNetFlowMatching, generate_samples, visualize_samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model (same as training)
    model = UNetFlowMatching(channels=3, base_channels=128, num_classes=10).to(device)
    
    # Try to load existing checkpoint
    try:
        checkpoint = torch.load('models/fm_unet_extended_final.pth', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded existing model successfully!")
    except:
        print("No existing model found. Train first with run_unet_cifar_extended.sbatch")
        return
    
    model.eval()
    
    # Create output directory
    os.makedirs("results/cifar", exist_ok=True)
    
    # Generate samples
    all_labels = torch.arange(10, device=device)
    
    for steps in [15, 25, 35]:
        print(f"Generating samples with {steps} steps...")
        
        with torch.no_grad():
            samples = generate_samples(model, device, all_labels, num_steps=steps)
        
        filename = f"results/cifar/existing_{steps}steps.png"
        visualize_samples(samples, all_labels, save_path=filename)
        
        del samples
        torch.cuda.empty_cache()
    
    print("Generation completed!")

if __name__ == "__main__":
    main()