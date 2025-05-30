import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import vgg16, VGG16_Weights

def get_my_vgg16_fasterrcnn_model(num_classes_to_set=2): # Added an argument for num_classes
    # --- 1. Load a pre-trained VGG16 model to use as the backbone ---
    vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    backbone = vgg_model.features
    backbone.out_channels = 512

    # --- 3. Define the Anchor Generator ---
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # --- 4. Define the RoI Pooler ---
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # --- 6. Create the Faster R-CNN model ---
    created_model = FasterRCNN(
        backbone,
        num_classes=num_classes_to_set, # Use the argument here
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    print(f"VGG16 Faster R-CNN model created with {num_classes_to_set} classes (incl. background).")
    return created_model

# This block runs ONLY if you execute model_setup.py directly
# (e.g., python model_setup.py)
# It will NOT run if you import get_my_vgg16_fasterrcnn_model from another file.
if __name__ == '__main__':
    print("--- Running model_setup.py directly for testing ---")
    num_classes = 2 # Example
    model = get_my_vgg16_fasterrcnn_model(num_classes_to_set=num_classes)

    # --- Test with a dummy input (Optional) ---
    try:
        dummy_image = torch.randn(1, 3, 300, 300)
        model.eval()
        with torch.no_grad():
            prediction = model(dummy_image)
        print("\nModel forward pass with dummy input successful.")
        if prediction:
            print(f"Dummy prediction output (first image):")
            print(f"  Boxes shape: {prediction[0]['boxes'].shape}")
            print(f"  Labels: {prediction[0]['labels']}")
            print(f"  Scores: {prediction[0]['scores']}")
    except Exception as e:
        print(f"\nError during model forward pass with dummy input: {e}")

    # --- Moving model to GPU (if available) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print(f"\nModel moved to {device} for testing.")
    else:
        device = torch.device("cpu")
        print(f"\nCUDA not available. Model remains on {device} for testing.")
    print("--- Finished model_setup.py direct run ---")