import torch
import torch.optim as optim
import time

# --- 1. Import from your other files ---
from model_setup import get_my_vgg16_fasterrcnn_model
from my_datasets import create_source_dataloader # We only need this for source training now
                                                # DetectionPresetTrain is used internally by create_source_dataloader
from generate_data import generate_all_synthetic_data # Our REAL data generator!

# --- Parameters for data generation and training ---
NUM_IMAGES_TO_GENERATE = 20 # Or however many you want for your experiment
NUM_CLASSES = 2 # 1 object class + background
BATCH_SIZE = 2 # Keep small for your GPU!
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# --- 2. Generate Your Data! ---
# This will create the images on disk and give us the lists we need.
# You only need to run this if you want to regenerate images or if they don't exist.
# If images already exist, this script will overwrite them.
# You could add logic to generate_data.py to skip if images exist and you want to reuse.
print("--- Calling Data Generation ---")
annotations_list_source, target_image_paths_list = generate_all_synthetic_data(
    num_images_to_generate=NUM_IMAGES_TO_GENERATE
)
print(f"Data generation complete. {len(annotations_list_source)} source images annotated.")
print(f"{len(target_image_paths_list)} target images generated.")


# --- 3. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Setup ---")
print(f"Using device: {device}")

# --- 4. Get Your Model ---
model = get_my_vgg16_fasterrcnn_model(num_classes_to_set=NUM_CLASSES)
model.to(device)
print("Successfully loaded VGG16 Faster R-CNN model and moved to device.")

# --- 5. Get Your Source DataLoader (using the data from generate_data.py) ---
source_dataloader = create_source_dataloader(
    ann_list=annotations_list_source, # From generate_data.py!
    batch_size_to_set=BATCH_SIZE,
    shuffle_data=True
)
print(f"Successfully created source_dataloader. Number of batches: {len(source_dataloader)}")

# --- 6. Optimizer ---
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

# --- 7. Training Loop (remains the same) ---
print(f"\n--- Starting Training for {NUM_EPOCHS} epochs ---")
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_epoch_loss = 0
    batch_count = 0

    for images, targets in source_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_epoch_loss += losses.item()
        batch_count += 1
    
    if batch_count > 0:
        avg_epoch_loss = total_epoch_loss / batch_count
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_duration:.2f}s, Average Loss: {avg_epoch_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - No data loaded from source_dataloader.")

print("\n--- Training Finished ---")
# Optional: save model
# model_save_path = 'my_vgg16_fasterrcnn_source_trained.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")