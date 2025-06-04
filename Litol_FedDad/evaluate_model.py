import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import random

# --- Import from your other project files ---
from model_setup import get_my_vgg16_fasterrcnn_model # To get the model architecture
from generate_data import generate_all_synthetic_data # To generate a few test images
# We don't strictly need my_datasets.py here if we handle image loading and transforms manually for inference

# --- Configuration ---
MODEL_PATH = 'my_vgg16_fasterrcnn_source_trained.pth' # Path to your saved model
NUM_CLASSES = 2 # 1 object class + background (must match how the model was trained)
NUM_SOURCE_TEST_IMAGES = 5 # How many new source-like images to test
NUM_TARGET_TEST_IMAGES = 5 # How many target images to test
SCORE_THRESHOLD = 0.5 # Only show predictions with confidence score above this
OUTPUT_DIR_EVAL = "evaluation_results"

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR_EVAL, exist_ok=True)

# --- 1. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Load the Trained Model ---
print(f"Loading model from: {MODEL_PATH}")
# First, create an instance of the model architecture
model = get_my_vgg16_fasterrcnn_model(num_classes_to_set=NUM_CLASSES)
# Then, load the saved weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Please train and save the model first.")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    print("Ensure the model architecture in model_setup.py matches the saved weights.")
    exit()

model.to(device)
model.eval() # Set the model to evaluation mode! This is crucial.
print("Model loaded successfully and set to evaluation mode.")

# --- 3. Prepare Test Images ---
# For simplicity, we'll generate a few new source-like images for testing.
# And use some of the existing target images.

# Generate NEW source test images (so the model hasn't seen them)
print(f"\nGenerating {NUM_SOURCE_TEST_IMAGES} new SOURCE test images...")
source_test_annotations, _ = generate_all_synthetic_data(
    num_images_to_generate=NUM_SOURCE_TEST_IMAGES,
    base_output_dir="synthetic_dataset_test_source" # Use a different dir to avoid overwriting
)
source_test_image_paths = [ann['image_path'] for ann in source_test_annotations]

# Get paths to existing TARGET images
# Assuming 'generate_data.py' created them in 'synthetic_dataset/target'
target_image_dir = "synthetic_dataset/target"
try:
    all_target_images = [os.path.join(target_image_dir, f) for f in os.listdir(target_image_dir) if f.endswith('.png')]
    if not all_target_images:
        print(f"No target images found in {target_image_dir}. Generating some for test.")
        _, target_test_image_paths = generate_all_synthetic_data(num_images_to_generate=NUM_TARGET_TEST_IMAGES, base_output_dir="synthetic_dataset") # This will use the default target dir
    else:
        target_test_image_paths = random.sample(all_target_images, min(NUM_TARGET_TEST_IMAGES, len(all_target_images)))
except FileNotFoundError:
    print(f"ERROR: Target image directory '{target_image_dir}' not found. Please generate target images first.")
    target_test_image_paths = []


# --- 4. Define a Transform for Inference ---
# For inference, we usually just need to convert the image to a tensor.
# Normalization might be needed if the model was trained with it.
# Our current training setup is simple, so ToTensor is the main step.
infer_transform = T.Compose([T.ToTensor()])

# --- 5. Prediction and Drawing Function ---
def predict_and_draw_boxes(image_path, model_to_use, device_to_use, transform_to_apply, threshold=0.5):
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"  Image not found: {image_path}")
        return None
    
    img_tensor = transform_to_apply(img_pil).to(device_to_use)
    
    with torch.no_grad(): # We don't need to calculate gradients for inference
        prediction = model_to_use([img_tensor]) # Model expects a batch of images

    # `prediction` is a list of dicts. For a single image, we take the first element.
    pred_boxes = prediction[0]['boxes'].cpu()
    pred_labels = prediction[0]['labels'].cpu()
    pred_scores = prediction[0]['scores'].cpu()

    # Filter predictions by score
    keep_indices = pred_scores > threshold
    final_boxes = pred_boxes[keep_indices]
    final_labels = pred_labels[keep_indices] # Label 1 is our object
    final_scores = pred_scores[keep_indices]

    # Draw boxes on the image
    draw = ImageDraw.Draw(img_pil)
    try:
        # You might need to specify a path to a .ttf font file for text to work well
        font = ImageFont.truetype("arial.ttf", 15) # Try a common font
    except IOError:
        font = ImageFont.load_default() # Fallback font

    for i in range(len(final_boxes)):
        box = final_boxes[i].tolist()
        label = final_labels[i].item()
        score = final_scores[i].item()

        # Our object is label 1. Label 0 is background (FasterRCNN usually doesn't output background as a prediction)
        if label == 1: # Only draw if it's our object class
            draw.rectangle(box, outline="lime", width=3)
            text = f"Obj: {score:.2f}"
            # Get text size using textbbox
            text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
            draw.rectangle(text_bbox, fill="lime")
            draw.text((box[0], box[1] - 15), text, fill="black", font=font)
        
    return img_pil

# --- 6. Run Evaluation ---
print(f"\n--- Evaluating on {len(source_test_image_paths)} SOURCE Test Images ---")
for i, img_path in enumerate(source_test_image_paths):
    print(f"Processing source test image: {img_path}...")
    result_img = predict_and_draw_boxes(img_path, model, device, infer_transform, SCORE_THRESHOLD)
    if result_img:
        save_path = os.path.join(OUTPUT_DIR_EVAL, f"source_test_pred_{i}.png")
        result_img.save(save_path)
        print(f"  Saved source prediction to {save_path}")

print(f"\n--- Evaluating on {len(target_test_image_paths)} TARGET Test Images ---")
for i, img_path in enumerate(target_test_image_paths):
    print(f"Processing target test image: {img_path}...")
    result_img = predict_and_draw_boxes(img_path, model, device, infer_transform, SCORE_THRESHOLD)
    if result_img:
        save_path = os.path.join(OUTPUT_DIR_EVAL, f"target_test_pred_{i}.png")
        result_img.save(save_path)
        print(f"  Saved target prediction to {save_path}")

print(f"\n--- Evaluation Finished ---")
print(f"Check the '{OUTPUT_DIR_EVAL}' folder for images with predictions.")

