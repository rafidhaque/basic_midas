import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. Configuration and Hyperparameters
# You can adjust these parameters
IMG_HEIGHT = 128  # Reduced image height for faster demo
IMG_WIDTH = 256   # Reduced image width for faster demo
NUM_CLASSES = 5   # Number of segmentation classes (Cityscapes has more, e.g., 19 for evaluation)
BATCH_SIZE = 4
EPOCHS = 5        # Number of training epochs, keep small for a quick demo
LEARNING_RATE = 0.001

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Synthetic Dataset for Demonstration
class SyntheticCityscapesDataset(Dataset):
    def __init__(self, num_samples, img_height, img_width, num_classes):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image (3 channels for RGB)
        # Values between 0 and 1
        image = np.random.rand(3, self.img_height, self.img_width).astype(np.float32)
        
        # Generate a random segmentation mask
        # Values between 0 and num_classes-1
        mask = np.random.randint(0, self.num_classes, (self.img_height, self.img_width)).astype(np.int64)
        
        return torch.from_numpy(image), torch.from_numpy(mask)

# Create synthetic datasets and dataloaders
# For a real scenario, you'd load your actual Cityscapes images and masks here
print("Creating synthetic datasets...")
train_dataset = SyntheticCityscapesDataset(num_samples=100, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_classes=NUM_CLASSES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = SyntheticCityscapesDataset(num_samples=20, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_classes=NUM_CLASSES)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Synthetic datasets created.")

# 3. Simple CNN Model for Semantic Segmentation (Basic FCN-like structure)
class SimpleSegmentationCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleSegmentationCNN, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Halves height and width
        )
        
        # Bottleneck/Bridge
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2) # Upsamples to original size
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function, and optimizer
print("Initializing model...")
model = SimpleSegmentationCNN(in_channels=3, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss() # Suitable for multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("Model initialized.")

# 4. Training Loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device) # Masks should be LongTensor for CrossEntropyLoss
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 10 == 0: # Print every 10 batches
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Training Loss: {epoch_loss:.4f}")

    # Simple validation (optional, just to see it run on val data)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Average Validation Loss: {avg_val_loss:.4f}")

print("Training finished.")

# 5. Quick Test (Optional: predict on a single synthetic image)
print("\nRunning a quick test on a synthetic image...")
model.eval()
with torch.no_grad():
    # Get a single batch from the validation loader
    test_images, test_masks = next(iter(val_loader))
    test_image_single = test_images[0:1].to(device) # Take the first image from the batch
    
    prediction = model(test_image_single)
    # The output 'prediction' will have shape [1, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH]
    # To get the predicted class for each pixel, take argmax along the channel dimension
    predicted_mask = torch.argmax(prediction, dim=1) 
    
    print(f"Input image shape: {test_image_single.shape}")
    print(f"Raw model output (logits) shape: {prediction.shape}")
    print(f"Predicted mask shape: {predicted_mask.shape}")
    print("Prediction for one image completed (predicted_mask contains class labels per pixel).")

print("\n---")
print("NOTE: This script uses SYNTHETIC (random) data for images and masks.")
print("To work with the actual Cityscapes dataset:")
print("1. You need to download Cityscapes (it requires registration).")
print("2. Preprocess the images and masks (e.g., resizing, normalization, converting masks to class indices).")
print("3. Implement a custom PyTorch `Dataset` to load your preprocessed Cityscapes data.")
print("   Replace `SyntheticCityscapesDataset` and its usage with your custom dataset.")
print("4. Adjust `NUM_CLASSES` and potentially image dimensions based on your preprocessing.")
print("---")
