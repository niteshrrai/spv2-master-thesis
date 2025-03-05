
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import matplotlib.patches as patches

def compute_bounding_box(keypoints, margin_factor, image_width, image_height):
    """Compute a square bounding box around 2D keypoints."""
    x_min, y_min = np.min(keypoints, axis=0)
    x_max, y_max = np.max(keypoints, axis=0)
    
    bbox_size = max(x_max - x_min, y_max - y_min) * margin_factor

    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2
    
    x_min, y_min = x_center - bbox_size / 2, y_center - bbox_size / 2
    x_max, y_max = x_center + bbox_size / 2, y_center + bbox_size / 2
    

    if x_min < 0: x_max, x_min = x_max - x_min, 0
    if y_min < 0: y_max, y_min = y_max - y_min, 0
    if x_max > image_width: x_min, x_max = x_min - (x_max - image_width), image_width
    if y_max > image_height: y_min, y_max = y_min - (y_max - image_height), image_height
    

    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image_width, x_max), min(image_height, y_max)
    

    final_size = min(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    
 
    x_min = max(0, x_center - final_size / 2)
    x_max = min(image_width, x_min + final_size)
    y_min = max(0, y_center - final_size / 2)
    y_max = min(image_height, y_min + final_size)
    
    return int(x_min), int(y_min), int(x_max), int(y_max)


def main():
    parser = argparse.ArgumentParser(description='Compute bounding boxes from 2D keypoints')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--json', type=str, required=True, help='JSON filename')
    parser.add_argument('--images', type=str, required=True, help='Images directory')
    parser.add_argument('--margin', type=float, default=1.2, help='Margin factor for bounding box size (default: 1.2)')
    parser.add_argument('--plot', action='store_true', help='Visualize bounding boxes')
    args = parser.parse_args()
    
    image_folder_path = Path(args.src) / "images"/ args.images
    json_path = Path(args.src) / args.json
    pose_data = json.load(open(json_path))
    
    for entry in pose_data:
        try:
            img_path = image_folder_path / entry['filename']
            with Image.open(img_path) as img:
                IMAGE_WIDTH, IMAGE_HEIGHT = img.size
                break
        except (FileNotFoundError, KeyError):
            continue
    
    print(f"Using image dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    
    plot_count = 0
    for entry in pose_data:
        try:
            keypoints = np.array(entry['keypoints_projected2D'])
            
            if keypoints.shape[0] == 2:
                keypoints = keypoints.T
            
            bbox = compute_bounding_box(keypoints, args.margin, IMAGE_WIDTH, IMAGE_HEIGHT)
            entry['bbox_gt'] = bbox
            
            # Only apply boundary check for visualization
            if args.plot and plot_count < 5:
                try:
                    img_path = image_folder_path / entry['filename']
                    image = Image.open(img_path).convert('RGB')
                    fig, ax = plt.subplots(1, figsize=(10, 8))
                    ax.imshow(np.array(image))
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    plt.title(entry['filename'])
                    plt.show()
                    plot_count += 1
                except FileNotFoundError:
                    print(f"Warning: {entry['filename']} not found for visualization")
        except (KeyError, ValueError) as e:
            print(f"Error processing entry: {e}")
    
    json.dump(pose_data, open(json_path, 'w'), indent=2)
    print("Processing complete. Bounding box co-ordinates added to JSON file.")

if __name__ == "__main__":
    main()


# Example Usage:
# python bbox.py --src subset --json train.json --images train --margin 1.2 --plot
# python bbox.py --src subset --json val.json --images val --margin 1.2 --plot
# python bbox.py --src subset --json test.json --images test --margin 1.2 --plot

