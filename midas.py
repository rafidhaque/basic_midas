import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation

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
        
        # Add GPU support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded successfully on: {device}")

        # --- 3. Prepare image for model ---
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to GPU if available
        for key, value in inputs.items():
            inputs[key] = value.to(device)

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
    image_input = "http://images.cocodataset.org/val2017/000000039769.jpg"
    selected_model = "Intel/dpt-large"

    print("--- MiDaS Depth Estimation Script ---")
    print("This script will download the model from Hugging Face Hub if it's not cached locally.")
    print("The first run might take several minutes depending on your internet speed and model size.")
    print("-------------------------------------\n")

    user_path = input(f"Enter image path or URL (or press Enter to use default: {image_input}): ")
    if user_path.strip():
        image_input = user_path.strip()

    estimate_depth(image_input, model_name=selected_model)