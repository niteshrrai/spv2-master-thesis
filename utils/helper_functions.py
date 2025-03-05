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
            if 'bbox_gt' in item and item['bbox_gt']:
                x_min, y_min, x_max, y_max = item['bbox_gt']
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

            if 'bbox_gt' in item and item['bbox_gt']:
                x_min, y_min, x_max, y_max = item['bbox_gt']
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


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2] format """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection_area = width * height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    return iou


def calculate_metrics(test_data):
    """Calculate and print evaluation metrics"""
    detections = [entry for entry in test_data if entry.get('bbox_pred') is not None]
    detection_rate = len(detections) / len(test_data) * 100
    
    ious = [entry.get('bbox_iou', 0) for entry in test_data if 'bbox_iou' in entry]
    
    if ious:
        avg_iou = sum(ious) / len(ious)
        iou_50 = sum(1 for iou in ious if iou >= 0.5) / len(ious) * 100
        iou_75 = sum(1 for iou in ious if iou >= 0.75) / len(ious) * 100
        
        print("\nEvaluation Metrics:")
        print(f"Detection Rate: {detection_rate:.2f}%")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"AP@0.5: {iou_50:.2f}%")
        print(f"AP@0.75: {iou_75:.2f}%")
    else:
        print("\nNo ground truth boxes available for evaluation")


def plot_bboxes_comparison(image_folder, json_file, num_images=5):
    """Comparison plot between ground truth and predicted bounding boxes"""

    data = json.load(open(json_file))
    valid_data = [item for item in data if 'bbox_gt' in item and 'bbox_pred' in item and item['bbox_pred'] is not None]
    
    if len(valid_data) == 0:
        print("No valid entries found with both ground truth and predictions")
        return
    
    random.shuffle(valid_data)
    selected_data = valid_data[:num_images]
    
    for item in selected_data:
        image_path = Path(image_folder) / item['filename']
        try:
            image = Image.open(image_path).convert('RGB')
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(np.array(image))
            
            if 'bbox_gt' in item and item['bbox_gt']:
                x_min, y_min, x_max, y_max = item['bbox_gt']
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                        linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth')
                ax.add_patch(rect)
            
            if 'bbox_pred' in item and item['bbox_pred']:
                x_min, y_min, x_max, y_max = item['bbox_pred']
                rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                        linewidth=2, edgecolor='r', facecolor='none', label='Prediction')
                ax.add_patch(rect)
            
            confidence = item.get('bbox_pred_conf', 'N/A')
            iou = item.get('bbox_iou', 'N/A')
            title = f"{item['filename']}"
            if isinstance(confidence, (int, float)):
                title += f"\nConfidence: {confidence:.2f}"
            if isinstance(iou, (int, float)):
                title += f", IoU: {iou:.2f}"
            
            plt.title(f"{item['filename']}")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
            
        except FileNotFoundError:
            print(f"Warning: {item['filename']} not found")