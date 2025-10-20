"""Simple script to view generated images."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_image(image_path):
    """Display an image if it exists."""
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Generated Results: {image_path}")
        plt.show()
        print(f"Displayed: {image_path}")
    else:
        print(f"Image not found: {image_path}")

def view_all_results():
    """View all generated result images."""
    image_files = [
        "specific_digits.png",
        "fm_samples.png", 
        "fm_cfortan_samples.png",
        "conditional_mnist_samples.png",
        "conditional_cifar_samples.png",
        "specific_cifar_classes.png"
    ]
    
    print("Available generated images:")
    for img_file in image_files:
        if os.path.exists(img_file):
            print(f"✓ {img_file}")
            view_image(img_file)
        else:
            print(f"✗ {img_file} (not generated yet)")

if __name__ == "__main__":
    print("=== Generated Image Viewer ===")
    view_all_results()