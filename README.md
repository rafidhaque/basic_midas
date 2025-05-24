# DPT Depth Estimation Script

This script performs monocular depth estimation using DPT (Dense Prediction Transformer) models from Hugging Face Transformers. It supports both local images and URLs, and visualizes the results using matplotlib.

## Features

- Supports multiple DPT models from Hugging Face Hub
- Handles local images and URLs
- Displays side-by-side comparison of original image and depth map
- Includes depth value normalization and visualization options
- Error handling for common issues

## Installation

pip install torch transformers pillow requests matplotlib numpy

*Note: Install [PyTorch](https://pytorch.org/) separately according to your system configuration.*

## Usage

python depth_estimation.py

When prompted, enter the path to an image file or an image URL, or press Enter to use the default example.
The script will download the model from Hugging Face Hub if not already cached.
The original image and the estimated depth map will be displayed in a matplotlib window.

## Example Output

--- MiDaS Depth Estimation Script ---
This script will download the model from Hugging Face Hub if it's not cached locally.
The first run might take several minutes depending on your internet speed and model size.
-------------------------------------

Enter image path or URL (or press Enter to use default: http://images.cocodataset.org/val2017/000000039769.jpg):
Image loaded from URL: http://images.cocodataset.org/val2017/000000039769.jpg
Loading model: Intel/dpt-large. This might take a moment...
Model loaded successfully.
Estimating depth...
Depth estimation complete.
Raw depth values range: Min=0.21, Max=15.74

## Configuration

You can change the default image and model by editing these lines in the script:

image_input = "path/to/your/image.jpg"  # Local path or URL
selected_model = "Intel/dpt-large"      # Or try "Intel/dpt-hybrid-midas"

Popular model options:
- "Intel/dpt-large" (default)
- "Intel/dpt-hybrid-midas"
- "Intel/dpt-beit-large-512"

## Output

- Left: Original image
- Right: Estimated depth map (normalized, grayscale)
- Colorbar: Depth scale (darker = farther, lighter = closer)

## Troubleshooting

- Ensure your image path or URL is correct and accessible.
- If the model fails to download, check your internet connection and Hugging Face access.
- For large images or models, ensure your system has enough RAM or use a GPU-enabled environment.

## References

- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/index
- MiDaS Project: https://github.com/isl-org/MiDaS
- DPT Paper: https://arxiv.org/abs/2103.13413

## License

MIT License. See LICENSE for details.

## Full Script

import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation # Updated import

def estimate_depth(image_path_or_url, model_name="Intel/dpt-large"):
    """
    Estimates depth from a single image using a DPT model (MiDaS-like).

    Args:
        image_path_or_url (str): Path to a local image file or URL of an image.
        model_name (str): Name of the DPT model from Hugging Face Model Hub.
                          Examples: "Intel/dpt-large", "Intel/dpt-hybrid-midas"
    """
    try:
        # --- 1. Load Image ---
        if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
            print(f"Image loaded from URL: {image_path_or_url}")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
            print(f"Image loaded from local path: {image_path_or_url}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    try:
        # --- 2. Load DPT Model and Processor ---
        print(f"Loading model: {model_name}. This might take a moment...")
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
        print("Model loaded successfully.")

        # --- 3. Prepare image for model ---
        inputs = processor(images=image, return_tensors="pt")

        # --- 4. Perform Depth Estimation ---
        print("Estimating depth...")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # --- 5. Process Output ---
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output_depth_map = prediction.cpu().numpy()

        min_val = np.min(output_depth_map)
        max_val = np.max(output_depth_map)
        if max_val - min_val > 1e-6:
            normalized_depth_map = (output_depth_map - min_val) / (max_val - min_val)
        else:
            normalized_depth_map = np.zeros_like(output_depth_map)

        print("Depth estimation complete.")
        print(f"Raw depth values range: Min={min_val:.2f}, Max={max_val:.2f}")

        # --- 6. Display Results ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        depth_display = axes[1].imshow(normalized_depth_map, cmap="gray")
        axes[1].set_title(f"Estimated Depth Map ({model_name})")
        axes[1].axis('off')

        fig.colorbar(depth_display, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        plt.suptitle("MiDaS-like Depth Estimation", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except Exception as e:
        print(f"An error occurred during model processing or display: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # --- Configuration ---
    # Option 1: Use a local image path
    # image_input = "path/to/your/image.jpg" # <--- CHANGE THIS TO YOUR IMAGE PATH

    # Option 2: Use an image URL (for quick testing)
    image_input = "http://images.cocodataset.org/val2017/000000039769.jpg" # Example COCO image URL
    # image_input = "https://hips.hearstapps.com/hmg-prod/images/large-cat-breed-1553197454.jpg" # Example cat image

    # You can choose different DPT models. Some popular ones:
    # "Intel/dpt-large" - General purpose, good quality.
    # "Intel/dpt-hybrid-midas" - Trained on a mix of datasets, often good for diverse scenes.
    # "Intel/dpt-beit-large-512" - A more recent BEiT-based DPT model.
    # "ainize/MiDaS-v2_1-large" - Though this might require slightly different handling if not directly DPT.
    # Check Hugging Face Model Hub for more "depth-estimation" models.
    selected_model = "Intel/dpt-large"
    # selected_model = "Intel/dpt-hybrid-midas"

    print("--- MiDaS Depth Estimation Script ---")
    print("This script will download the model from Hugging Face Hub if it's not cached locally.")
    print("The first run might take several minutes depending on your internet speed and model size.")
    print("-------------------------------------\n")

    user_path = input(f"Enter image path or URL (or press Enter to use default: {image_input}): ")
    if user_path.strip():
        image_input = user_path.strip()

    estimate_depth(image_input, model_name=selected_model)
