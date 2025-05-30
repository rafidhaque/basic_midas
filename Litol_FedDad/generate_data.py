from PIL import Image, ImageDraw
import os
import random

# --- Configuration ---
output_dir_source = "synthetic_dataset/source"
output_dir_target = "synthetic_dataset/target" # We'll make target images look different
num_images_to_generate = 20 # Let's start with a small number
image_width = 300
image_height = 300
background_color_source = "white"
shape_color_source = "red"

# For target domain, let's make them visually different
background_color_target = "gray"
shape_color_target = "blue" # Different shape color for target
add_noise_to_target = True # Optional: add simple noise

# Ensure output directories exist
os.makedirs(output_dir_source, exist_ok=True)
os.makedirs(output_dir_target, exist_ok=True)

# This will store our annotations: list of dictionaries
# Each dict: {'image_path': path, 'boxes': [[x1,y1,x2,y2], ...], 'labels': [1, ...]}
annotations_list_source = []
# We don't generate training annotations for the target domain in this setup,
# but you'd need labeled target test images for evaluation later.

print(f"Generating {num_images_to_generate} SOURCE images in '{output_dir_source}'...")
for i in range(num_images_to_generate):
    # Create a new image with a white background
    img = Image.new('RGB', (image_width, image_height), background_color_source)
    draw = ImageDraw.Draw(img)

    current_image_boxes = []
    current_image_labels = []

    # Let's draw 1 to 3 shapes per image
    num_shapes = random.randint(1, 3)

    for _ in range(num_shapes):
        # Randomly determine shape size (between 30x30 and 100x100 pixels)
        shape_w = random.randint(30, 100)
        shape_h = random.randint(30, 100)

        # Randomly determine top-left position (x1, y1)
        # Ensure the shape fits within the image
        x1 = random.randint(0, image_width - shape_w)
        y1 = random.randint(0, image_height - shape_h)

        # Calculate bottom-right position (x2, y2)
        x2 = x1 + shape_w
        y2 = y1 + shape_h

        # These coordinates ARE our bounding box!
        box = [x1, y1, x2, y2]

        # Draw the rectangle
        draw.rectangle(box, fill=shape_color_source, outline="black") # Added an outline for visibility

        # Store the annotation for this shape
        current_image_boxes.append(box)
        current_image_labels.append(1) # Label '1' for our object class

    # Save the image
    image_filename = f"source_image_{i:03d}.png"
    image_path = os.path.join(output_dir_source, image_filename)
    img.save(image_path)

    # Add to our annotations list
    annotations_list_source.append({
        'image_path': image_path,
        'boxes': current_image_boxes,
        'labels': current_image_labels
    })

print(f"\nGenerated {num_images_to_generate} SOURCE images.")
print("Example SOURCE annotations (first 3 images):")
for k in range(min(3, len(annotations_list_source))):
    print(annotations_list_source[k])


# --- Now let's generate TARGET images (visually different, UNLABELED for training) ---
# We'll reuse the same logic but change colors and optionally add noise.
# For FedDAD, the target domain is unlabeled during training.
# We generate them here just to have the image files.
print(f"\nGenerating {num_images_to_generate} TARGET images in '{output_dir_target}'...")
for i in range(num_images_to_generate):
    img = Image.new('RGB', (image_width, image_height), background_color_target)
    draw = ImageDraw.Draw(img)

    num_shapes = random.randint(1, 3)
    for _ in range(num_shapes):
        shape_w = random.randint(30, 100)
        shape_h = random.randint(30, 100)
        x1 = random.randint(0, image_width - shape_w)
        y1 = random.randint(0, image_height - shape_h)
        x2 = x1 + shape_w
        y2 = y1 + shape_h
        draw.rectangle([x1, y1, x2, y2], fill=shape_color_target, outline="darkgray")

    # Optional: Add simple noise to target images
    if add_noise_to_target:
        pixels = img.load() # Get the pixel map
        for r_idx in range(img.width):
            for c_idx in range(img.height):
                noise = random.randint(-20, 20) # Adjust noise level
                r, g, b = pixels[r_idx, c_idx]
                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                b = max(0, min(255, b + noise))
                pixels[r_idx, c_idx] = (r, g, b)

    image_filename = f"target_image_{i:03d}.png"
    image_path = os.path.join(output_dir_target, image_filename)
    img.save(image_path)

print(f"\nGenerated {num_images_to_generate} TARGET images.")
print("Remember: Target images are typically UNLABELED for domain adaptation training.")
print("You would need a separate labeled TARGET TEST SET for final evaluation.")

