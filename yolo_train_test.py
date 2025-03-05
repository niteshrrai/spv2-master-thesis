import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from utils.helper_functions import calculate_iou, calculate_metrics


def yolo_train(config_path, model_path="yolo11n.pt", epochs=100, batch_size=16, img_size=640, device=None):
    """Trains a YOLO model with the specified parameters and returns the Path to the best model's weights"""
    
    model = YOLO(model_path)
    results = model.train(
        data=config_path,
        epochs=epochs,        # Number of training epochs
        imgsz=img_size,       # Image size for training
        batch=batch_size,     # Batch size
        device=device,        # Training device (None for auto-selection, [0,1] for multiple GPUs)
        plots=True,           # Generate training plots
        save=True,            # Save model checkpoints
        seed=42,              # Random seed for reproducibility
        optimizer='auto',     # Optimizer selection (SGD, Adam, etc.)
        save_period=10,       # Save checkpoint every N epochs
        patience=20,          # Early stopping patience
        cos_lr=True,          # Use cosine learning rate schedule
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate ratio
        val=True,             # Run validation during training
        degrees=10.0,         # Random rotation augmentation (±10°)
        translate=0.1,        # Random translation augmentation (±10%)
        scale=0.5             # Random scaling augmentation (0.5-1.5x)
    )
    return str(Path(results.save_dir) / 'weights' / 'best.pt')


def yolo_inference(model_path, test_json, image_dir, save_results=False):
    """Run YOLO inference on test images and add predicted bounding boxes to test.json"""

    model = YOLO(model_path)

    with open(test_json, 'r') as f:
        test_data = json.load(f)
    print(f"Running inference on {len(test_data)} test images...")
    
    for i, entry in enumerate(test_data):
        filename = entry['filename']
        image_path = Path(image_dir) / filename
        
        if not image_path.exists():
            print(f"Warning: Image {filename} not found, skipping...")
            continue
        
        results = model.predict(
            source=str(image_path),
            max_det=1,  # Only detect one object (Tango satellite)
            save=save_results,
            project=os.path.dirname(test_json) if save_results else None,
            name="results" if save_results else None,
            exist_ok=True
        )
        
        if len(results) > 0:
            result = results[0] 
            
            if len(result.boxes) > 0:
                boxes = result.boxes
                conf_values = boxes.conf.cpu().numpy()
                best_idx = conf_values.argmax()
                
                xyxy = boxes.xyxy.cpu().numpy()[best_idx]
                x1, y1, x2, y2 = map(int, xyxy)

                entry['bbox_pred'] = [x1, y1, x2, y2]
                entry['bbox_pred_conf'] = float(conf_values[best_idx])
                
                if 'bbox_gt' in entry:
                    gt_box = entry['bbox_gt']
                    iou = calculate_iou(gt_box, [x1, y1, x2, y2])
                    entry['bboxes_iou'] = iou
            else:
                entry['bbox_pred'] = None
                entry['bbox_pred_conf'] = 0.0
                if 'bbox_gt' in entry:
                    entry['bboxes_iou'] = 0.0
        
        if (i + 1) % 10 == 0 or i == len(test_data) - 1:
            print(f"Processed {i + 1}/{len(test_data)} images")
    
    with open(test_json, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Inference complete. Updated {test_json} with predictions.")
    calculate_metrics(test_data)


def main():
    parser = argparse.ArgumentParser(description='YOLO Train and Test Script')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train-test'], required=True, help='Operation mode: train, test, or train-test')
    # Training arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--model_path', type=str, default='yolo11n.pt', help='Initial model path for training or trained model path for testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--device', type=str, default=None, help='Training device (None for auto, 0 for single GPU, [0,1] for multiple GPUs)')
    # Testing arguments
    parser.add_argument('--json', type=str, default='test.json', help='Test JSON filename')
    parser.add_argument('--images', type=str, default='images/test', help='Test images directory')
    parser.add_argument('--save', action='store_true', help='Save annotated images with bounding boxes')
    
    args = parser.parse_args()
    src_dir = os.path.abspath(args.src)
    best_model_path = None
    
    # Training mode
    if args.mode in ['train', 'train-test']:
        if not args.config:
            raise ValueError("--config is required for training mode")
            
        config_path = os.path.join(src_dir, args.config)
        print(f"Training with config: {config_path}")
        
        best_model_path = yolo_train(
            config_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device
        )
        
        print(f"Training complete. Best model saved at: {best_model_path}")
    
    # Testing mode
    if args.mode in ['test', 'train-test']:
        model_path = best_model_path if best_model_path else args.model_path
        json_path = os.path.join(src_dir, args.json)
        image_dir = os.path.join(src_dir, args.images)
        
        if not os.path.exists(json_path):
            print(f"Error: Test JSON file not found at {json_path}")
            return
        
        if not os.path.exists(image_dir):
            print(f"Error: Test images directory not found at {image_dir}")
            return
        
        print(f"Running inference with model: {model_path}")
        print(f"Test JSON: {json_path}")
        print(f"Image directory: {image_dir}")
        
        yolo_inference(
            model_path=model_path,
            test_json=json_path,
            image_dir=image_dir,
            save_results=args.save
        )

if __name__ == "__main__":
    main()


# Example Usage:
# Train only: python yolo_train_test.py --src subset --mode train --config subset.yaml --model_path yolo11n.pt --epochs 10
# Test only: python yolo_train_test.py --src subset --mode test --json test.json --images images/test --model_path runs/detect/train/weights/best.pt
# Train and test: python yolo_train_test.py --src subset --mode train-test --config subset.yaml --json test.json --images images/test --epochs 10 --save