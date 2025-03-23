import os
import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from efficientnet_pytorch import EfficientNet
from keypoint_dataset import KeypointDataset, create_data_loaders, get_transforms


class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=11, pretrained=True):
        """
        Initialize the keypoint detection model using EfficientNet as backbone
        
        Args:
            num_keypoints (int): Number of keypoints to predict
            pretrained (bool): Whether to use pretrained weights
        """
        super(KeypointModel, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        
        # Get the number of features from the backbone
        num_features = self.backbone._fc.in_features
        
        # Replace the classifier with keypoint regressor
        self.backbone._fc = nn.Identity()
        
        # Keypoint regression head
        self.keypoint_regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_keypoints * 2)  # x,y coordinates for each keypoint
        )
        
    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.keypoint_regressor(features)
        batch_size = keypoints.shape[0]
        return keypoints.view(batch_size, -1, 2)  # Reshape to (batch_size, num_keypoints, 2)


def train_step(model, dataloader, criterion, optimizer, device):
    """
    Run one training epoch
    
    Args:
        model: The neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run the training on
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        images = batch['image'].to(device)
        target_keypoints = batch['keypoints'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_keypoints = model(images)
        
        # Compute loss
        loss = criterion(pred_keypoints, target_keypoints)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(dataloader)


def val_step(model, dataloader, criterion, device):
    """
    Run validation
    
    Args:
        model: The neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run the validation on
        
    Returns:
        tuple: (average validation loss, average keypoint error)
    """
    model.eval()
    running_loss = 0.0
    keypoint_errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            # Get data
            images = batch['image'].to(device)
            target_keypoints = batch['keypoints'].to(device)
            
            # Forward pass
            pred_keypoints = model(images)
            
            # Compute loss
            loss = criterion(pred_keypoints, target_keypoints)
            running_loss += loss.item()
            
            # Compute mean per joint position error (MPJPE)
            error = torch.sqrt(((pred_keypoints - target_keypoints) ** 2).sum(dim=2)).mean(dim=1)
            keypoint_errors.extend(error.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    avg_keypoint_error = np.mean(keypoint_errors)
    
    return avg_loss, avg_keypoint_error


def train_model(model, train_loader, val_loader, args):
    """
    Train the keypoint detection model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Training arguments
        
    Returns:
        str: Path to the best model weights
    """
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    print(f"Training on: {device}")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    

    # Early stopping setup
    early_stopping_patience = 20
    no_improve_count = 0

    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create log file
    log_path = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"Epoch,Train Loss,Val Loss,Val Error,Time (s),Learning Rate\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        # Train
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_error = val_step(model, val_loader, criterion, device)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.6f}")
            no_improve_count = 0  # Reset counter
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

        epoch_time = time.time() - start_time
        print(f"Epoch: {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val Error: {val_error:.4f} pixels | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Log metrics
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_error:.4f},{epoch_time:.2f}\n,{current_lr:.6f}\n")
    
    return best_model_path


def test_model(model, test_loader, output_dir, device):
    """
    Test the keypoint detection model and visualize results
    
    Args:
        model: Trained model
        test_loader: Test data loader
        output_dir: Directory to save results
        device: Device to run inference on
    """
    model.eval()
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    keypoint_errors = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Get data
            images = batch['image'].to(device)
            target_keypoints = batch['keypoints'].to(device)
            filenames = batch['filename']
            bboxes = batch['bbox'].numpy()
            crop_sizes = batch['crop_size'].numpy()
            
            # Forward pass
            pred_keypoints = model(images)
            
            # Calculate error
            error = torch.sqrt(((pred_keypoints - target_keypoints) ** 2).sum(dim=2)).mean(dim=1)
            keypoint_errors.extend(error.cpu().numpy())
            
            # Store predictions
            for i in range(len(filenames)):
                pred = pred_keypoints[i].cpu().numpy()
                target = target_keypoints[i].cpu().numpy()
                bbox = bboxes[i]
                crop_size = crop_sizes[i]
                
                # Scale keypoints back to original image coordinates
                pred_original = rescale_to_original(pred, bbox, crop_size)
                target_original = rescale_to_original(target, bbox, crop_size)
                
                all_predictions.append({
                    'filename': filenames[i],
                    'bbox': bbox.tolist(),
                    'pred_keypoints': pred_original.tolist(),
                    'gt_keypoints': target_original.tolist(),
                    'error': error[i].item()
                })
                
                # Visualize every 20th sample
                if batch_idx % 20 == 0 and i == 0:
                    visualize_prediction(
                        filenames[i], 
                        pred, 
                        target, 
                        os.path.join(results_dir, f"pred_{batch_idx}.png")
                    )
    
    # Save all predictions
    with open(os.path.join(output_dir, 'test_predictions.json'), 'w') as f:
        json.dump(all_predictions, f, indent=2)

    # Calculate PCK (Percentage of Correct Keypoints)
    thresholds = [5, 10, 15, 20]  # Pixel thresholds
    pck_results = calculate_pck(all_predictions, thresholds)
    
    # Print test metrics
    mean_error = np.mean(keypoint_errors)
    median_error = np.median(keypoint_errors)
    print(f"\nTest Results:")
    print(f"Mean Keypoint Error: {mean_error:.4f} pixels")
    print(f"Median Keypoint Error: {median_error:.4f} pixels")
    
    for threshold in thresholds:
        print(f"PCK@{threshold}: {pck_results[threshold]:.4f}")

    # Save test metrics
    with open(os.path.join(output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Mean Keypoint Error: {mean_error:.4f} pixels\n")
        f.write(f"Median Keypoint Error: {median_error:.4f} pixels\n")
        for threshold in thresholds:
            f.write(f"PCK@{threshold}: {pck_results[threshold]:.4f}\n")

def calculate_pck(predictions, thresholds):
    """
    Calculate Percentage of Correct Keypoints (PCK) at different thresholds
    Args:
        predictions: List of prediction dicts
        thresholds: List of distance thresholds in pixels
    Returns:
        dict: PCK values at each threshold
    """
    results = {t: 0 for t in thresholds}
    total_keypoints = 0
    
    for pred in predictions:
        pred_keypoints = np.array(pred['pred_keypoints'])
        gt_keypoints = np.array(pred['gt_keypoints'])
        
        # Calculate distances for each keypoint
        distances = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=1))
        
        # Count correct keypoints at each threshold
        for t in thresholds:
            results[t] += np.sum(distances < t)
        
        total_keypoints += len(distances)
    
    # Normalize by total number of keypoints
    for t in thresholds:
        results[t] /= total_keypoints
    
    return results


def visualize_prediction(filename, pred_keypoints, gt_keypoints, save_path=None):
    """
    Visualize predicted keypoints vs ground truth
    
    Args:
        filename: Image filename
        pred_keypoints: Predicted keypoints
        gt_keypoints: Ground truth keypoints
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ground truth keypoints
    plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='g', marker='o', label='Ground Truth')
    
    # Plot predicted keypoints
    plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c='r', marker='x', label='Prediction')
    
    # Add title and legend
    plt.title(f"Keypoint Prediction: {filename}")
    plt.legend()
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def rescale_to_original(keypoints, bbox, crop_size):
    """
    Rescale keypoints from crop coordinates back to original image coordinates
    
    Args:
        keypoints: Keypoints in crop coordinates
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        crop_size: Size of the cropped image [width, height]
        
    Returns:
        numpy.ndarray: Keypoints in original image coordinates
    """
    x_min, y_min = bbox[0], bbox[1]
    crop_width, crop_height = crop_size
    
    # Scale factor from model output (224x224) to crop size
    scale_x = crop_width / 224
    scale_y = crop_height / 224
    
    # First scale to crop size
    keypoints_scaled = keypoints.copy()
    keypoints_scaled[:, 0] *= scale_x
    keypoints_scaled[:, 1] *= scale_y
    
    # Then add the bbox offset
    keypoints_scaled[:, 0] += x_min
    keypoints_scaled[:, 1] += y_min
    
    return keypoints_scaled


def inference(model, image_path, output_path=None, bbox=None):
    """
    Run inference on a single image
    
    Args:
        model: Trained model
        image_path: Path to the input image
        output_path: Path to save the visualization
        bbox: Optional bounding box, if None will use the entire image
    """
    device = next(model.parameters()).device
    transforms = get_transforms()[1]  # Use validation transforms
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    if bbox is None:
        # Use the entire image
        w, h = image.size
        bbox = [0, 0, w, h]
    
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    crop_width, crop_height = cropped_image.size
    
    # Apply transforms
    cropped_np = np.array(cropped_image)
    transformed = transforms(image=cropped_np)
    tensor_image = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        pred_keypoints = model(tensor_image)[0].cpu().numpy()
    
    # Rescale keypoints
    pred_original = rescale_to_original(
        pred_keypoints, 
        bbox, 
        np.array([crop_width, crop_height])
    )
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(np.array(image))
    ax.set_title("Predicted Keypoints")
    
    # Draw bounding box
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                             linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    
    # Draw keypoints
    ax.scatter(pred_original[:, 0], pred_original[:, 1], c='r', marker='x', s=40)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train and Test Keypoint Detection')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train-test', 'inference'], 
                        required=True, help='Operation mode')
    parser.add_argument('--src', type=str, required=True, help='Source directory')
    parser.add_argument('--output-dir', type=str, default='runs/keypoints', 
                        help='Output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device or -1 for CPU')
    
    # Model parameters
    parser.add_argument('--num-keypoints', type=int, default=11, help='Number of keypoints')
    parser.add_argument('--model-path', type=str, help='Path to model weights')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')
    
    # Inference parameters
    parser.add_argument('--image', type=str, help='Image path for inference')
    parser.add_argument('--bbox', type=str, help='Bounding box for inference (x_min,y_min,x_max,y_max)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    
    # Create model
    model = KeypointModel(num_keypoints=args.num_keypoints)
    
    # Load model weights if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    model = model.to(device)
    
    if args.mode in ['train', 'train-test']:
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            argparse.Namespace(
                src=args.src,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        )
        
        print(f"Starting training for {args.epochs} epochs...")
        best_model_path = train_model(model, train_loader, val_loader, args)
        print(f"Training complete. Best model saved at: {best_model_path}")
        
        # Update model path and load best model
        args.model_path = best_model_path
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    if args.mode in ['test', 'train-test']:
        if args.mode == 'test' and not args.model_path:
            raise ValueError("Model path must be provided for test mode")
        
        print("Creating test data loader...")
        _, _, test_loader = create_data_loaders(
            argparse.Namespace(
                src=args.src,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        )
        
        print(f"Starting testing...")
        test_model(model, test_loader, args.output_dir, device)
        print(f"Testing complete. Results saved to: {args.output_dir}")
    
    if args.mode == 'inference':
        if not args.model_path:
            raise ValueError("Model path must be provided for inference mode")
        
        if not args.image:
            raise ValueError("Image path must be provided for inference mode")
        
        bbox = None
        if args.bbox:
            bbox = [int(x) for x in args.bbox.split(',')]
        
        output_path = os.path.join(args.output_dir, 'inference_result.png')
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"Running inference on: {args.image}")
        inference(model, args.image, output_path, bbox)
        print(f"Inference complete. Result saved to: {output_path}")


if __name__ == "__main__":
    main()

# Example Usage:
# Train: python keypoint_detection.py --mode train --src subset --output-dir runs/keypoints --epochs 1
# Test: python keypoint_detection.py --mode test --src subset --model-path runs/keypoints/best_model.pth
# Train and Test: python keypoint_detection.py --mode train-test --src subset --output-dir runs/keypoints --epochs 50
# Inference: python keypoint_detection.py --mode inference --src subset --model-path runs/keypoints/best_model.pth --image subset/images/test/img001583.jpg --bbox 703,374,1163,834