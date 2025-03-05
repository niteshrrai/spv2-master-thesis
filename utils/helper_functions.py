import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def organize_synthetic_images(base_path="synthetic"):
    """Used to split image folder into train and val using their respective json files."""
    os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "val"), exist_ok=True)
    
    for split in ["train", "val"]:
        with open(os.path.join(base_path, f"{split}.json")) as f:
            files = {entry["filename"] for entry in json.load(f)}
        
        for file in files:
            shutil.move(os.path.join(base_path, "images", file), os.path.join(base_path, split, file))


def consolidate_images(base_path):
    images_path = Path(base_path) / "images"
    os.makedirs(images_path, exist_ok=True)
    
    for subset in ["train", "val", "test"]:
        src, dst = Path(base_path) / subset, images_path / subset
        if src.exists():
            os.makedirs(dst, exist_ok=True)
            for file in src.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(dst / file.name))
            try:
                os.rmdir(src)
            except OSError:
                print(f"Skipped deleting {src}, as it is not empty.")


def plot_keypoints(image_folder, json_file, num_images=5):
    data = json.load(open(json_file))
    random.shuffle(data)
    
    for item in data[:num_images]:
        image_path = Path(image_folder) / item['filename']
        try:
            image = Image.open(image_path).convert('RGB')
            points2D = np.array(item['keypoints_projected2D']).T
            plt.figure(figsize=(10, 8))
            visible_mask = (points2D[1, :] >= 0) & (points2D[1, :] < 1200) & (points2D[0, :] >= 0) & (points2D[0, :] < 1920)
            visible_points = points2D[:, visible_mask]
            plt.imshow(image)
            plt.scatter(visible_points[0, :], visible_points[1, :], c='r', marker='x', s=40)
            plt.title(f"{item['filename']}")
            plt.show()
        except FileNotFoundError:
            print(f"Warning: {item['filename']} not found")

def plot_bounding_boxes(image_folder, json_file, num_images=5):
    data = json.load(open(json_file))
    random.shuffle(data)
    
    for item in data[:num_images]:
        image_path = Path(image_folder) / item['filename']
        try:
            image = Image.open(image_path).convert('RGB')
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(np.array(image))
            if 'bbox' in item and item['bbox']:
                x_min, y_min, x_max, y_max = item['bbox']
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                         linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
            plt.title(f"{item['filename']}")
            plt.show()      
        except FileNotFoundError:
            print(f"Warning: {item['filename']} not found")


def plot_keypoints_bbox(image_folder, json_file, num_images=5):
    data = json.load(open(json_file))
    random.shuffle(data)
    
    for item in data[:num_images]:
        image_path = Path(image_folder) / item['filename']
        try:
            image = Image.open(image_path).convert('RGB')
            points2D = np.array(item['keypoints_projected2D']).T
            
            plt.figure(figsize=(10, 8))
            plt.imshow(image)

            visible_mask = (points2D[1, :] >= 0) & (points2D[1, :] < 1200) & (points2D[0, :] >= 0) & (points2D[0, :] < 1920)
            visible_points = points2D[:, visible_mask]
            plt.scatter(visible_points[0, :], visible_points[1, :], c='r', marker='x', s=40)

            if 'bbox' in item and item['bbox']:
                x_min, y_min, x_max, y_max = item['bbox']
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                         linewidth=2, edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)

            plt.title(f"{item['filename']}")
            plt.show()
            
        except FileNotFoundError:
            print(f"Warning: {item['filename']} not found")



def plot_yolo_bbox(image_folder, label_folder, num_images=5):
    image_folder = Path(image_folder)
    label_folder = Path(label_folder)
    
    label_files = sorted(list(label_folder.glob('*.txt')))
    random.shuffle(label_files)
    sample_files = label_files[:num_images]
    
    for label_file in sample_files:
        image_path = image_folder / f"{label_file.stem}.jpg"
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
    
        plt.figure(figsize=(10, 8))
        plt.imshow(image)

        for line in lines:
            values = list(map(float, line.strip().split()))
            x_center, y_center = values[1], values[2]
            width, height = values[3], values[4]
            
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            
            rect = patches.Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='g', facecolor='None')
            plt.gca().add_patch(rect)
        
        plt.title(f"{image_path.name}")
        plt.tight_layout()
        plt.show()
