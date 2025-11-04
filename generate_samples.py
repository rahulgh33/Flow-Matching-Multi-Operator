"""
Generate samples from trained Flow Matching models.

Usage:
    python generate_samples.py --model models/fm_unet_extended_final.pth --steps 25
"""

import torch
import torch.nn as nn
import argparse
import os
import sys

# Add conditional directory to path
sys.path.append('conditional')
from fm_cifar_conditional import UNetFlowMatching, generate_samples, visualize_samples, generate_specific_classes

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    # Create model with saved config
    model = UNetFlowMatching(
        channels=config['channels'],
        base_channels=config['base_channels'], 
        num_classes=config['num_classes']
    ).to(device)
    
    # Load model weights (already contains EMA weights if available)
    print("Loading model weights (with EMA applied during training)...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Trained for {checkpoint['epoch']} epochs, final loss: {checkpoint['loss']:.4f}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained Flow Matching model')
    parser.add_argument('--model', type=str, default='models/fm_unet_extended_final.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--steps', type=int, nargs='+', default=[15, 25, 35],
                       help='Number of integration steps for generation')
    parser.add_argument('--output_dir', type=str, default='results/cifar',
                       help='Output directory for generated images')
    parser.add_argument('--specific_classes', type=int, nargs='+', default=[0, 3, 5, 8],
                       help='Specific classes to generate (airplane=0, cat=3, dog=5, ship=8)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        print("Train a model first using run_unet_cifar_extended.sbatch")
        return
    
    model = load_model(args.model, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples at different step counts
    all_labels = torch.arange(10, device=device)  # One example per class
    
    for steps in args.steps:
        print(f"Generating samples with {steps} steps...")
        
        with torch.no_grad():
            samples = generate_samples(model, device, all_labels, num_steps=steps)
        
        filename = os.path.join(args.output_dir, f"extended_{steps}steps.png")
        visualize_samples(samples, all_labels, save_path=filename)
        
        # Clear memory
        del samples
        torch.cuda.empty_cache()
    
    # Generate specific classes
    print(f"Generating specific classes: {args.specific_classes}...")
    with torch.no_grad():
        generate_specific_classes(model, device, args.specific_classes)
    
    print("Generation completed successfully!")
    print(f"Check {args.output_dir}/ for generated images")

if __name__ == "__main__":
    main()