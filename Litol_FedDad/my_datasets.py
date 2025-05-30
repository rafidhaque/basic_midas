from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os

# --- Ensure 'annotations_list_source' and 'target_image_paths' are available ---
# This section tries to ensure the script is runnable by creating dummy data
# if the actual generated data isn't found or passed.
# In your workflow, you'd ensure these are populated from your image generation step.

if not os.path.exists('synthetic_dataset/source') or not os.path.exists('synthetic_dataset/target'):
    print("WARNING: Synthetic dataset directories not found. Creating dummy data for script execution.")
    if not os.path.exists('synthetic_dataset/source'): os.makedirs('synthetic_dataset/source')
    if not os.path.exists('synthetic_dataset/target'): os.makedirs('synthetic_dataset/target')
    
    dummy_img_s = Image.new('RGB', (100,100), 'white')
    dummy_img_s_path = 'synthetic_dataset/source/dummy_source_000.png'
    dummy_img_s.save(dummy_img_s_path)
    annotations_list_source = [{'image_path': dummy_img_s_path, 'boxes': [[10,10,50,50]], 'labels': [1]}]
    
    dummy_img_t = Image.new('RGB', (100,100), 'gray')
    dummy_img_t_path = 'synthetic_dataset/target/dummy_target_000.png'
    dummy_img_t.save(dummy_img_t_path)
    target_image_paths = [dummy_img_t_path]
else:
    # Attempt to load/regenerate if not in global scope (for standalone running of this snippet)
    if 'annotations_list_source' not in globals() or not annotations_list_source:
        print("Regenerating a small sample of annotations_list_source for example...")
        from PIL import ImageDraw 
        annotations_list_source = []
        for i in range(max(2, len(os.listdir('synthetic_dataset/source')) if os.path.exists('synthetic_dataset/source') else 2)): # Ensure at least 2 for batching
            img_s_path = os.path.join('synthetic_dataset/source', f'source_image_example_{i}.png')
            if not os.path.exists(img_s_path): # Create if doesn't exist
                 img_s = Image.new('RGB', (300, 300), 'white')
                 draw_s = ImageDraw.Draw(img_s)
                 x1,y1,x2,y2 = 50+i*10, 50+i*10, 100+i*10, 100+i*10
                 draw_s.rectangle([x1,y1,x2,y2], fill='red')
                 img_s.save(img_s_path)
            annotations_list_source.append({'image_path': img_s_path, 'boxes': [[50+i*10, 50+i*10, 100+i*10, 100+i*10]], 'labels': [1]})
    
    if 'target_image_paths' not in globals() or not target_image_paths:
        target_image_paths = [os.path.join('synthetic_dataset/target', f) for f in os.listdir('synthetic_dataset/target') if f.endswith('.png')]
        if not target_image_paths: # Create dummy if empty
            print("No target images found, creating dummy target image.")
            dummy_img_t = Image.new('RGB', (100,100), 'gray')
            dummy_img_t_path = 'synthetic_dataset/target/dummy_target_000.png'
            dummy_img_t.save(dummy_img_t_path)
            target_image_paths.append(dummy_img_t_path)


class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5):
        # For simplicity, we'll focus on ToTensor.
        # More complex augmentations would require careful handling of bounding boxes.
        if data_augmentation == "hflip":
            # Note: T.RandomHorizontalFlip needs to be applied to both image and boxes.
            # This basic setup doesn't include box flipping.
            # self.transforms = T.Compose([T.RandomHorizontalFlip(p=hflip_prob), T.ToTensor()])
            self.transforms = T.Compose([T.ToTensor()]) # Keeping it simple
        else: # Default: just convert to tensor
            self.transforms = T.Compose([T.ToTensor()])

    def __call__(self, img, target): # target can be None
        # Apply the composed transforms (e.g., ToTensor) to the image
        img = self.transforms(img)
        
        # If geometric augmentations were part of self.transforms (e.g., flip, resize),
        # the 'target' (bounding boxes) would need to be adjusted here accordingly.
        # Since we are only using ToTensor for now, target boxes are not affected geometrically.
        return img, target # Return transformed image and original (or potentially modified) target


class SyntheticObjectDetectionDataset(Dataset):
    def __init__(self, annotation_list, transforms=None):
        self.annotation_list = annotation_list
        self.transforms = transforms

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        img_annotation = self.annotation_list[idx]
        img_path = img_annotation['image_path']
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"ERROR: Source image not found at {img_path}. Returning black image.")
            img = Image.new("RGB", (300, 300), "black")

        boxes = torch.as_tensor(img_annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(img_annotation['labels'], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if boxes.shape[0] > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.empty((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target) # Assumes transform handles img and target

        return img, target


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths_list, transforms=None):
        self.image_paths_list = image_paths_list
        self.transforms = transforms # This will be an instance of DetectionPresetTrain

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, idx):
        img_path = self.image_paths_list[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"ERROR: Target image not found at {img_path}. Returning black image.")
            img = Image.new("RGB", (300, 300), "black")

        if self.transforms:
            # If self.transforms is our DetectionPresetTrain, it expects (img, target).
            # We pass None for target as this is an unlabeled dataset.
            # DetectionPresetTrain.__call__ will handle target being None.
            img, _ = self.transforms(img, None) # Call with img and None for target
            # The underscore _ receives the second return value (which would be None)
        return img


# --- Example Usage (should be in your main script or a testing block) ---
if __name__ == '__main__': # Ensures this runs only when script is executed directly
    # 1. Ensure 'annotations_list_source' and 'target_image_paths' are populated
    # This might involve running your image generation script or loading saved data.
    # The dummy data generation at the top of the file handles this for standalone testing.
    print(f"Number of source annotations: {len(annotations_list_source)}")
    print(f"Number of target image paths: {len(target_image_paths)}")


    # 2. Create instances of your Datasets
    source_transform = DetectionPresetTrain(data_augmentation="none")
    source_dataset = SyntheticObjectDetectionDataset(
        annotations_list_source,
        transforms=source_transform
    )

    target_transform = DetectionPresetTrain(data_augmentation="none") # Can be the same or different
    target_unlabeled_dataset = UnlabeledImageDataset(
        target_image_paths,
        transforms=target_transform
    )

    # 3. Create DataLoaders
    source_dataloader = torch.utils.data.DataLoader(
        source_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    target_unlabeled_dataloader = torch.utils.data.DataLoader(
        target_unlabeled_dataset,
        batch_size=2,
        shuffle=False # Usually False for unlabeled/test data
    )

    print(f"\nCreated Source Dataset with {len(source_dataset)} samples.")
    print(f"Created Source DataLoader with batch_size=2.")

    try:
        images_batch, targets_batch = next(iter(source_dataloader))
        print(f"\nSuccessfully fetched a batch from Source DataLoader!")
        print(f"  Number of images in batch: {len(images_batch)}")
        print(f"  Image 0 type: {type(images_batch[0])}, Image 0 shape: {images_batch[0].shape}")
        print(f"  Number of targets in batch: {len(targets_batch)}")
        print(f"  Target 0 example keys: {targets_batch[0].keys()}")
    except Exception as e:
        print(f"\nError fetching data from Source DataLoader: {e}")

    print(f"\nCreated Unlabeled Target Dataset with {len(target_unlabeled_dataset)} samples.")
    print(f"Created Unlabeled Target DataLoader with batch_size=2.")

    try:
        target_images_batch = next(iter(target_unlabeled_dataloader))
        print(f"\nSuccessfully fetched a batch from Unlabeled Target DataLoader!")
        print(f"  Number of images in batch: {len(target_images_batch)}")
        print(f"  Image 0 type: {type(target_images_batch[0])}, Image 0 shape: {target_images_batch[0].shape}")
    except Exception as e:
        print(f"\nError fetching data from Unlabeled Target DataLoader: {e}")
