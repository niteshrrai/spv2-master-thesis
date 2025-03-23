import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import argparse
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KeypointDataset(Dataset):

    def __init__(self, src_dir, json_file, images_dir, split, transform=None):
        """
        Initialize the KeypointDataset
        
        Args:
            src_dir (str): Source directory containing data
            json_file (str): JSON filename with keypoint data
            images_dir (str): Directory containing images
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.src_dir = Path(src_dir)
        self.json_path = self.src_dir / json_file
        self.image_dir = self.src_dir / "images" / images_dir
        self.split = split
        self.transform = transform

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        for entry in self.data:
            try:
                img_path = self.image_dir / entry['filename']
                if img_path.exists():
                    with Image.open(img_path) as img:
                        self.image_width, self.image_height = img.size
                        break
            except (FileNotFoundError, KeyError):
                continue
        
        self.data = [entry for entry in self.data if self._is_valid_entry(entry)]
        print(f"Loaded {len(self.data)} valid entries for {split} split")
    
    def _is_valid_entry(self, entry):
        if 'filename' not in entry or 'keypoints_projected2D' not in entry:
            return False
        
        if self.split in ['train', 'val'] and 'bbox_gt' not in entry:
            return False
            
        if self.split == 'test' and 'bbox_pred_sq' not in entry:
            return False
        
        img_path = self.image_dir / entry['filename']
        if not img_path.exists():
            return False
        
        return True
    
    def __len__(self):
        return len(self.data)
    
    def _get_bbox(self, entry):
        if self.split == 'test':
            return entry['bbox_pred_sq']
        else:
            return entry['bbox_gt']
    
    def _rescaled_keypoints(self, keypoints, bbox):
        """Rescale keypoints from original image coordinates to crop coordinates"""
        keypoints_array = np.array(keypoints, dtype=np.float32)
        x_min, y_min = bbox[0], bbox[1]
        
        rescaled = keypoints_array.copy()
        rescaled[:, 0] -= x_min
        rescaled[:, 1] -= y_min
        
        return rescaled
    
    def __getitem__(self, idx):

        entry = self.data[idx]
        
        img_path = self.image_dir / entry['filename']
        image = Image.open(img_path).convert('RGB')
        
        bbox = self._get_bbox(entry)
        x_min, y_min, x_max, y_max = bbox
        
        keypoints = entry['keypoints_projected2D']
        
        rescaled_keypoints = self._rescaled_keypoints(keypoints, bbox)
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        crop_width, crop_height = cropped_image.size
        
        cropped_image_np = np.array(cropped_image)
        
        if self.transform:
            transformed = self.transform(image=cropped_image_np, keypoints=rescaled_keypoints)
            cropped_image = transformed['image']
            rescaled_keypoints = np.array(transformed['keypoints'], dtype=np.float32)
        
        sample = {
            'image': cropped_image,
            'keypoints': torch.tensor(rescaled_keypoints, dtype=torch.float32),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'filename': entry['filename'],
            'crop_size': torch.tensor([crop_width, crop_height], dtype=torch.float32)
        }
        
        return sample


def get_transforms():
    """Returns augmentation pipelines for training and validation."""
    train_transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.Affine(shear=15, scale=(0.9, 1.1), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=0, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            
            A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
            
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.CLAHE(p=0.8),
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Posterize(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
            
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.CLAHE(p=0.8),
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Posterize(p=0.8),
                A.Blur(p=0.8),
            ], p=1.0),
            
            A.Normalize(
                mean=[0.4897, 0.4897, 0.4897],
                std=[0.2330, 0.2330, 0.2330],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], 
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    
    val_transforms = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0.4897, 0.4897, 0.4897],
                std=[0.2330, 0.2330, 0.2330],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], 
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    
    return train_transforms, val_transforms


def create_data_loaders(args):
    """Create train, validation, and test data loaders."""

    train_transforms, val_transforms = get_transforms()
    
    # Create datasets
    train_dataset = KeypointDataset(
        src_dir=args.src,
        json_file='train.json',
        images_dir='train',
        split='train',
        transform=train_transforms
    )
    
    val_dataset = KeypointDataset(
        src_dir=args.src,
        json_file='val.json',
        images_dir='val',
        split='val',
        transform=val_transforms
    )
    
    test_dataset = KeypointDataset(
        src_dir=args.src,
        json_file='test.json',
        images_dir='test',
        split='test',
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def visualize_dataset_samples(dataset, num_samples=5):
    """Visualize random samples from the dataset, including original, cropped and augmented versions"""
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        entry = dataset.data[idx]
        img_path = dataset.image_dir / entry['filename']
        image = Image.open(img_path).convert('RGB')
        bbox = dataset._get_bbox(entry)
        keypoints = entry['keypoints_projected2D']

        rescaled_keypoints = dataset._rescaled_keypoints(keypoints, bbox)

        x_min, y_min, x_max, y_max = bbox
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        sample = dataset[idx]
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image with bbox and keypoints
        ax[0].imshow(np.array(image))
        ax[0].set_title(f"Original: {entry['filename']}")

        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=2, edgecolor='g', facecolor='none')
        ax[0].add_patch(rect)
        keypoints_array = np.array(keypoints)
        ax[0].scatter(keypoints_array[:, 0], keypoints_array[:, 1], c='r', marker='x', s=40)
        
        # Cropped image with keypoints
        ax[1].imshow(np.array(cropped_image))
        ax[1].set_title(f"Cropped Image: {entry['filename']}")

        ax[1].scatter(rescaled_keypoints[:, 0], rescaled_keypoints[:, 1], c='r', marker='x', s=40)
        
        # Augmented cropped image with keypoints
        if isinstance(sample['image'], torch.Tensor):
            aug_img = sample['image'].permute(1, 2, 0).numpy()
            mean = np.array([0.4897, 0.4897, 0.4897])
            std = np.array([0.2330, 0.2330, 0.2330])
            aug_img = std * aug_img + mean
            aug_img = np.clip(aug_img, 0, 1)
        else:
            aug_img = sample['image']   

        ax[2].imshow(aug_img)
        ax[2].set_title("Augmented Image")

        aug_keypoints = sample['keypoints'].numpy()
        ax[2].scatter(aug_keypoints[:, 0], aug_keypoints[:, 1], c='r', marker='x', s=40)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Keypoint Dataset Visualization Tool')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True, help='Dataset split')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--plot', action='store_true', help='Visualize random samples')
    args = parser.parse_args()
    
    train_transforms, val_transforms = get_transforms()
    
    if args.split == 'train':
        dataset = KeypointDataset(
            src_dir=args.src,
            json_file='train.json',
            images_dir='train',
            split='train',
            transform=train_transforms
        )
    elif args.split == 'val':
        dataset = KeypointDataset(
            src_dir=args.src,
            json_file='val.json',
            images_dir='val',
            split='val',
            transform=val_transforms
        )
    else:  # test
        dataset = KeypointDataset(
            src_dir=args.src,
            json_file='test.json',
            images_dir='test',
            split='test',
            transform=val_transforms
        )
    
    if args.plot:
        visualize_dataset_samples(dataset, args.samples)
    
    train_loader, val_loader, test_loader = create_data_loaders(args)
    
    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()

# Example Usage:
# python keypoint_dataset.py --src subset --split train --samples 5 --plot
# python keypoint_dataset.py --src subset --split val --samples 5 --plot
# python keypoint_dataset.py --src subset --split test --samples 10 --plot
