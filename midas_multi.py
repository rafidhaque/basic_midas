import torch
from PIL import Image
import numpy as np
import os
from pathlib import Path
from transformers import DPTImageProcessor, DPTForDepthEstimation

def create_depth_maps_batch(input_folder, output_folder, model_name="Intel/dpt-large"):
    """
    Creates depth maps for all images in a folder and saves them to another folder.
    
    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path to folder where depth maps will be saved
        model_name (str): Name of the DPT model from Hugging Face Model Hub
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files from input folder
    input_path = Path(input_folder)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not image_files:
        print(f"No supported image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    try:
        # Load DPT Model and Processor once
        print(f"Loading model: {model_name}...")
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
        
        # Add GPU support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model loaded successfully on: {device}")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            try:
                print(f"Processing {i}/{len(image_files)}: {image_file.name}")
                
                # Load image
                image = Image.open(image_file).convert("RGB")
                
                # Prepare image for model
                inputs = processor(images=image, return_tensors="pt")
                
                # Move inputs to GPU if available
                for key, value in inputs.items():
                    inputs[key] = value.to(device)
                
                # Perform depth estimation
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                # Process output
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                output_depth_map = prediction.cpu().numpy()
                
                # Normalize depth map to 0-255 range for saving
                min_val = np.min(output_depth_map)
                max_val = np.max(output_depth_map)
                if max_val - min_val > 1e-6:
                    normalized_depth_map = (output_depth_map - min_val) / (max_val - min_val)
                else:
                    normalized_depth_map = np.zeros_like(output_depth_map)
                
                # Convert to 8-bit image
                depth_image = (normalized_depth_map * 255).astype(np.uint8)
                
                # Save depth map
                output_filename = f"depth_{image_file.stem}.png"
                output_path = os.path.join(output_folder, output_filename)
                
                # Convert numpy array to PIL Image and save
                depth_pil = Image.fromarray(depth_image, mode='L')
                depth_pil.save(output_path)
                
                print(f"  ✓ Depth map saved: {output_filename}")
                
            except Exception as e:
                print(f"  ✗ Error processing {image_file.name}: {e}")
                continue
                
    except Exception as e:
        print(f"Error loading model or during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nBatch processing complete!")

def main():
    # Base directory
    base_dir = r"C:\Users\rafha\OneDrive\Documents\basic_midas"
    
    # Define input and output folders
    input_folder = os.path.join(base_dir, "input_images")
    output_folder = os.path.join(base_dir, "depth_maps")
    
    # Create input folder if it doesn't exist
    os.makedirs(input_folder, exist_ok=True)
    
    print("--- Batch MiDaS Depth Estimation Script ---")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("This script will process all images in the input folder.")
    print("Supported formats: JPG, JPEG, PNG, BMP, TIFF")
    print("-" * 50)
    
    # Check if input folder has images
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    input_path = Path(input_folder)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not image_files:
        print(f"\nNo images found in {input_folder}")
        print("Please add some images to the input folder and run the script again.")
        input("Press Enter to exit...")
        return
    
    # Ask user to confirm
    response = input(f"\nFound {len(image_files)} images. Proceed with depth map generation? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Process images
    create_depth_maps_batch(input_folder, output_folder)
    
    print(f"\nAll depth maps have been saved to: {output_folder}")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
