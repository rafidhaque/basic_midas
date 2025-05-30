from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os # Still useful for os.path.join if needed, but not for dummy data generation

# --- Dataset Classes (These definitions remain the same as your last working version) ---
class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5):
        if data_augmentation == "hflip":
            self.transforms = T.Compose([T.ToTensor()]) 
        else: 
            self.transforms = T.Compose([T.ToTensor()])
    def __call__(self, img, target): 
        img = self.transforms(img)
        return img, target

class SyntheticObjectDetectionDataset(Dataset):
    def __init__(self, annotation_list, transforms=None):
        self.annotation_list = annotation_list # Expect this to be provided
        self.transforms = transforms
    def __len__(self):
        return len(self.annotation_list)
    def __getitem__(self, idx):
        img_annotation = self.annotation_list[idx]
        img_path = img_annotation['image_path']
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # print(f"Warning (Dataset): Image not found at {img_path}. Returning black image.")
            img = Image.new("RGB", (300, 300), "black") # Default size
        boxes = torch.as_tensor(img_annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(img_annotation['labels'], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if boxes.numel() > 0: # Check if boxes tensor is not empty
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.empty((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths_list, transforms=None):
        self.image_paths_list = image_paths_list # Expect this to be provided
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths_list)
    def __getitem__(self, idx):
        img_path = self.image_paths_list[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # print(f"Warning (Dataset): Image not found at {img_path}. Returning black image.")
            img = Image.new("RGB", (300, 300), "black") # Default size
        if self.transforms:
            img, _ = self.transforms(img, None)
        return img
# --- End of Dataset Classes ---


# --- Functions to create DataLoaders (These definitions also remain the same) ---
def create_source_dataloader(ann_list, batch_size_to_set, shuffle_data=True):
    transform = DetectionPresetTrain(data_augmentation="none")
    dataset = SyntheticObjectDetectionDataset(ann_list, transforms=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_to_set,
        shuffle=shuffle_data,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    # print(f"Created Source DataLoader with {len(dataset)} samples, batch_size={batch_size_to_set}.")
    return dataloader

def create_target_unlabeled_dataloader(img_paths, batch_size_to_set, shuffle_data=False):
    transform = DetectionPresetTrain(data_augmentation="none")
    dataset = UnlabeledImageDataset(img_paths, transforms=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_to_set,
        shuffle=shuffle_data
    )
    # print(f"Created Target Unlabeled DataLoader with {len(dataset)} samples, batch_size={batch_size_to_set}.")
    return dataloader

# This block runs ONLY if you execute my_datasets.py directly
if __name__ == '__main__':
    print("--- Running my_datasets.py directly for testing ---")
    # To test this file directly, we'd need to call the data generation first
    # Or create some minimal dummy data here just for the test.
    
    # Example: Create minimal dummy data for testing this file's functions
    print("Creating minimal dummy data for my_datasets.py direct run test...")
    dummy_source_dir = "temp_test_data/source"
    dummy_target_dir = "temp_test_data/target"
    os.makedirs(dummy_source_dir, exist_ok=True)
    os.makedirs(dummy_target_dir, exist_ok=True)

    test_annotations_list = []
    for i in range(2):
        p = os.path.join(dummy_source_dir, f"test_s_{i}.png")
        Image.new('RGB', (50,50), 'pink').save(p)
        test_annotations_list.append({'image_path': p, 'boxes': [[10,10,20,20]], 'labels': [1]})

    test_target_paths = []
    for i in range(2):
        p = os.path.join(dummy_target_dir, f"test_t_{i}.png")
        Image.new('RGB', (50,50), 'lightblue').save(p)
        test_target_paths.append(p)
    
    print(f"Number of test source annotations: {len(test_annotations_list)}")
    print(f"Number of test target image paths: {len(test_target_paths)}")

    if test_annotations_list:
        test_source_loader = create_source_dataloader(test_annotations_list, batch_size_to_set=1)
        try:
            s_imgs, s_tgts = next(iter(test_source_loader))
            print(f"Successfully fetched a batch from test_source_loader (direct run). Img shape: {s_imgs[0].shape}")
        except Exception as e:
            print(f"Error fetching from test_source_loader (direct run): {e}")
    
    if test_target_paths:
        test_target_loader = create_target_unlabeled_dataloader(test_target_paths, batch_size_to_set=1)
        try:
            t_imgs = next(iter(test_target_loader))
            print(f"Successfully fetched a batch from test_target_loader (direct run). Img shape: {t_imgs[0].shape}")
        except Exception as e:
            print(f"Error fetching from test_target_loader (direct run): {e}")
    print("--- Finished my_datasets.py direct run ---")