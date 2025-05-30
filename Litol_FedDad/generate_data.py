from PIL import Image, ImageDraw
import os
import random

def generate_all_synthetic_data(num_images_to_generate=20, image_width=300, image_height=300, base_output_dir="synthetic_dataset"):
    output_dir_source = os.path.join(base_output_dir, "source")
    output_dir_target = os.path.join(base_output_dir, "target")

    background_color_source = "white"
    shape_color_source = "red"
    background_color_target = "gray"
    shape_color_target = "blue"
    add_noise_to_target = True

    os.makedirs(output_dir_source, exist_ok=True)
    os.makedirs(output_dir_target, exist_ok=True)

    generated_annotations_list_source = []
    generated_target_image_paths = []

    print(f"--- Running Data Generation (generate_data.py) ---")
    print(f"Generating {num_images_to_generate} SOURCE images in '{output_dir_source}'...")
    for i in range(num_images_to_generate):
        img = Image.new('RGB', (image_width, image_height), background_color_source)
        draw = ImageDraw.Draw(img)
        current_image_boxes = []
        current_image_labels = []
        num_shapes = random.randint(1, 3)

        for _ in range(num_shapes):
            shape_w = random.randint(30, 100)
            shape_h = random.randint(30, 100)
            x1 = random.randint(0, image_width - shape_w)
            y1 = random.randint(0, image_height - shape_h)
            x2 = x1 + shape_w
            y2 = y1 + shape_h
            box = [x1, y1, x2, y2]
            draw.rectangle(box, fill=shape_color_source, outline="black")
            current_image_boxes.append(box)
            current_image_labels.append(1)

        image_filename = f"source_image_{i:03d}.png"
        image_path = os.path.join(output_dir_source, image_filename)
        img.save(image_path)
        generated_annotations_list_source.append({
            'image_path': image_path,
            'boxes': current_image_boxes,
            'labels': current_image_labels
        })

    print(f"Generated {num_images_to_generate} SOURCE images.")

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

        if add_noise_to_target:
            pixels = img.load()
            for r_idx in range(img.width):
                for c_idx in range(img.height):
                    noise = random.randint(-20, 20)
                    r, g, b = pixels[r_idx, c_idx]
                    pixels[r_idx, c_idx] = (max(0, min(255, r + noise)), max(0, min(255, g + noise)), max(0, min(255, b + noise)))

        image_filename = f"target_image_{i:03d}.png"
        image_path = os.path.join(output_dir_target, image_filename)
        img.save(image_path)
        generated_target_image_paths.append(image_path)

    print(f"Generated {num_images_to_generate} TARGET images.")
    print(f"--- Finished Data Generation (generate_data.py) ---")
    
    return generated_annotations_list_source, generated_target_image_paths

# This block runs if you execute generate_data.py directly
if __name__ == '__main__':
    print("Running generate_data.py directly to create images...")
    # You can specify how many images you want when testing it directly
    annotations_source, target_paths = generate_all_synthetic_data(num_images_to_generate=5) 
    print("\nExample SOURCE annotations from direct run (first 2):")
    for k in range(min(2, len(annotations_source))):
        print(annotations_source[k])
    print("\nExample TARGET paths from direct run (first 2):")
    for k in range(min(2, len(target_paths))):
        print(target_paths[k])