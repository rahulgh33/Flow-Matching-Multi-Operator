"""Demo script for conditional Flow Matching generation."""

import torch
from fm_mnist_conditional import ConditionalFlowMatchingNet, generate_specific_digits
from fm_cifar_conditional import ConditionalFlowMatchingNetCIFAR, generate_specific_classes, CIFAR10_CLASSES


def demo_mnist_conditional():
    """Demo conditional MNIST generation."""
    print("=== MNIST Conditional Generation Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or create a model (for demo, we'll create untrained model)
    model = ConditionalFlowMatchingNet(channels=1, num_classes=10).to(device)
    
    print("Generating specific digits: 0, 1, 2, 3...")
    generate_specific_digits(model, device, [0, 1, 2, 3])
    
    print("Generating all digits 0-9...")
    generate_specific_digits(model, device, list(range(10)))


def demo_cifar_conditional():
    """Demo conditional CIFAR-10 generation."""
    print("\n=== CIFAR-10 Conditional Generation Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or create a model (for demo, we'll create untrained model)
    model = ConditionalFlowMatchingNetCIFAR(channels=3, num_classes=10).to(device)
    
    print("Available CIFAR-10 classes:")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        print(f"  {i}: {class_name}")
    
    print("\nGenerating specific classes: airplane, cat, dog...")
    generate_specific_classes(model, device, [0, 3, 5])  # airplane, cat, dog


def interactive_generation():
    """Interactive generation where user can specify what to generate."""
    print("\n=== Interactive Generation ===")
    
    while True:
        print("\nChoose dataset:")
        print("1. MNIST (digits 0-9)")
        print("2. CIFAR-10 (objects)")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("Enter digits to generate (0-9), separated by commas:")
            print("Example: 1,3,7,9")
            digits_input = input("Digits: ").strip()
            
            try:
                digits = [int(d.strip()) for d in digits_input.split(",")]
                digits = [d for d in digits if 0 <= d <= 9]  # Filter valid digits
                
                if digits:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = ConditionalFlowMatchingNet(channels=1, num_classes=10).to(device)
                    generate_specific_digits(model, device, digits)
                else:
                    print("No valid digits entered!")
                    
            except ValueError:
                print("Invalid input! Please enter digits separated by commas.")
        
        elif choice == "2":
            print("Available classes:")
            for i, class_name in enumerate(CIFAR10_CLASSES):
                print(f"  {i}: {class_name}")
            
            print("Enter class indices (0-9), separated by commas:")
            print("Example: 0,3,5 (airplane, cat, dog)")
            classes_input = input("Classes: ").strip()
            
            try:
                class_indices = [int(c.strip()) for c in classes_input.split(",")]
                class_indices = [c for c in class_indices if 0 <= c <= 9]  # Filter valid classes
                
                if class_indices:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = ConditionalFlowMatchingNetCIFAR(channels=3, num_classes=10).to(device)
                    generate_specific_classes(model, device, class_indices)
                else:
                    print("No valid class indices entered!")
                    
            except ValueError:
                print("Invalid input! Please enter class indices separated by commas.")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")


if __name__ == "__main__":
    print("Flow Matching Conditional Generation Demo")
    print("=" * 50)
    
    # Note: These demos use untrained models, so outputs will be noise
    # To get good results, you need to train the models first
    print("Note: This demo uses untrained models for demonstration.")
    print("To get meaningful results, train the models first using:")
    print("  python fm_mnist_conditional.py")
    print("  python fm_cifar_conditional.py")
    
    demo_mnist_conditional()
    demo_cifar_conditional()
    
    # Uncomment for interactive mode
    # interactive_generation()