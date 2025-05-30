import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import vgg16, VGG16_Weights

# --- 1. Load a pre-trained VGG16 model to use as the backbone ---
# We'll use weights pre-trained on ImageNet
# VGG16_Weights.IMAGENET1K_V1 is one of the available pre-trained weights
vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# --- 2. Extract the feature extraction part of VGG16 ---
# The paper uses VGG16. In PyTorch, the 'features' part of VGG16
# consists of its convolutional layers. This will serve as our backbone.
# The output of the VGG16 features part has 512 channels.
backbone = vgg_model.features
backbone.out_channels = 512 # Last conv layer in VGG16 features has 512 output channels

# --- 3. Define the Anchor Generator for the Region Proposal Network (RPN) ---
# These are standard values, but you can tune them later if needed.
# AnchorGenerator makes anchors at different sizes and aspect ratios.
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), # Anchor sizes (pixels)
    aspect_ratios=((0.5, 1.0, 2.0),) # Aspect ratios
)

# --- 4. Define the RoI (Region of Interest) Pooler ---
# This module extracts features for each proposed region from the feature maps.
# It takes the backbone's output and the proposed boxes.
# 'featmap_names' specifies which feature map from the backbone to use.
# For VGG16's 'features' module, the output is a single feature map,
# conventionally named '0' if you pass it directly.
# If backbone was a FeaturePyramidNetwork (FPN), it would have multiple levels.
# Since we are using VGG16's features directly, it outputs a single scale feature map.
# The RoIAlign output size is (7,7) for each RoI.
roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0'], # Assumes the output of backbone.features is named '0'
                         # This might need adjustment if using a more complex backbone wrapper
    output_size=7,
    sampling_ratio=2
)

# --- 5. Define the number of classes ---
# IMPORTANT: This number must include the background class.
# So, if you want to detect 1 type of object (e.g., "cat"), num_classes = 2 (1 for cat + 1 for background).
# If you want to detect 3 types of objects (e.g., "cat", "dog", "car"), num_classes = 4.
# Let's say we want to detect 1 custom object type.
num_classes = 2 # e.g., 1 object class + 1 background class

# --- 6. Create the Faster R-CNN model ---
# We combine the backbone, RPN (via anchor_generator), RoI pooler, and specify num_classes.
# The box_predictor component of FasterRCNN will be automatically created
# with the correct number of output features for `num_classes`.
model = FasterRCNN(
    backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

print("Faster R-CNN model with VGG16 backbone created successfully!")
print(f"Number of output classes (including background): {num_classes}")

# --- 7. Test with a dummy input (Optional) ---
# This helps to ensure the model architecture is correctly connected.
# Create a dummy image tensor: (batch_size, channels, height, width)
# Batch size 1, 3 color channels (RGB), 300x300 pixels (adjust as needed for your data)
# Ensure the input size is appropriate for VGG16 (it can handle various sizes, but
# very small sizes might not work well with its downsampling)
try:
    dummy_image = torch.randn(1, 3, 300, 300)
    model.eval() # Set model to evaluation mode for inference
    with torch.no_grad(): # No need to calculate gradients for this test
        prediction = model(dummy_image)
    print("\nModel forward pass with dummy input successful.")
    # `prediction` will be a list of dictionaries, one for each image in the batch.
    # Each dictionary contains 'boxes', 'labels', 'scores'.
    if prediction:
        print(f"Dummy prediction output (first image):")
        print(f"  Boxes shape: {prediction[0]['boxes'].shape}")
        print(f"  Labels: {prediction[0]['labels']}")
        print(f"  Scores: {prediction[0]['scores']}")
except Exception as e:
    print(f"\nError during model forward pass with dummy input: {e}")
    print("This might indicate an issue with layer dimensions or connections.")
    print("Common issues: RoIAlign featmap_names, backbone output channels, or input image size.")

# --- 8. Moving model to GPU (if available) ---
# This is crucial for training.
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print(f"\nModel moved to {device}")
    # You would also need to move your input data to the GPU during training/inference:
    # dummy_image = dummy_image.to(device)
else:
    device = torch.device("cpu")
    print(f"\nCUDA not available. Model remains on {device}.")
    print("Training will be very slow on CPU.")

