"""
Prediction script for dog heart VHS prediction model.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import csv
from tqdm import tqdm

from data import create_dataloaders
from models.vit_regressor import DogHeartViTWithAttention
from models.utils import load_model
from utils.logger import setup_logger
from utils.metrics import evaluate_model
from utils.visualization import visualize_test_predictions


def predict_test_set(model, test_loader, output_path=None, device='cuda'):
    """
    Make predictions on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        output_path (str, optional): Path to save predictions
        device (str): Device to run model on
        
    Returns:
        dict: Dictionary mapping filenames to predictions
    """
    model.eval()
    predictions = {}
    
    # Create progress bar
    pbar = tqdm(test_loader, desc="Predicting on test set")
    
    with torch.no_grad():
        for batch in pbar:
            # Get data
            images = batch['image'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            
            # Extract predictions
            vhs_preds = outputs['vhs'].cpu().numpy().flatten()
            
            # Store predictions with filenames
            for i, filename in enumerate(filenames):
                predictions[filename] = {
                    'vhs': float(vhs_preds[i])
                }
                
                # Add points if available
                if 'points' in outputs:
                    points = outputs['points'][i].cpu().numpy()
                    predictions[filename]['points'] = points.tolist()
    
    # Save predictions to CSV if output_path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine fields based on what's available
        sample_pred = next(iter(predictions.values()))
        has_points = 'points' in sample_pred
        
        if has_points:
            # Create CSV with VHS and flattened points
            with open(output_path, 'w', newline='') as csvfile:
                num_points = len(sample_pred['points'])
                point_headers = []
                for i in range(num_points):
                    point_headers.extend([f'point_{i+1}_x', f'point_{i+1}_y'])
                
                fieldnames = ['filename', 'vhs'] + point_headers
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for filename, pred_dict in predictions.items():
                    row = {'filename': filename, 'vhs': pred_dict['vhs']}
                    
                    # Add flattened points
                    points = pred_dict['points']
                    for i, point in enumerate(points):
                        row[f'point_{i+1}_x'] = point[0]
                        row[f'point_{i+1}_y'] = point[1]
                    
                    writer.writerow(row)
        else:
            # Create simple CSV with just VHS
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'vhs']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for filename, pred_dict in predictions.items():
                    writer.writerow({
                        'filename': filename,
                        'vhs': pred_dict['vhs']
                    })
        
        print(f"Predictions saved to {output_path}")
    
    # Return predictions
    return predictions


def analyze_predictions(predictions):
    """
    Analyze predictions.
    
    Args:
        predictions (dict): Dictionary mapping filenames to predictions
        
    Returns:
        dict: Dictionary of statistics
    """
    # Extract VHS values
    vhs_values = [pred['vhs'] for pred in predictions.values()]
    
    # Calculate statistics
    stats = {
        'count': len(vhs_values),
        'mean': float(np.mean(vhs_values)),
        'std': float(np.std(vhs_values)),
        'min': float(np.min(vhs_values)),
        'max': float(np.max(vhs_values)),
        'median': float(np.median(vhs_values)),
        '25th_percentile': float(np.percentile(vhs_values, 25)),
        '75th_percentile': float(np.percentile(vhs_values, 75))
    }
    
    # If points are available, analyze them too
    sample_pred = next(iter(predictions.values()))
    if 'points' in sample_pred:
        # Get all points
        all_points = np.array([pred['points'] for pred in predictions.values()])
        
        # Calculate centroid statistics
        centroids = np.mean(all_points, axis=1)
        
        stats['centroid_x_mean'] = float(np.mean(centroids[:, 0]))
        stats['centroid_y_mean'] = float(np.mean(centroids[:, 1]))
        stats['centroid_x_std'] = float(np.std(centroids[:, 0]))
        stats['centroid_y_std'] = float(np.std(centroids[:, 1]))
        
        # Calculate perimeters
        perimeters = []
        for points in all_points:
            perimeter = 0
            for i in range(len(points)):
                next_i = (i + 1) % len(points)
                p1 = points[i]
                p2 = points[next_i]
                dist = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
                perimeter += dist
            perimeters.append(perimeter)
        
        stats['perimeter_mean'] = float(np.mean(perimeters))
        stats['perimeter_std'] = float(np.std(perimeters))
        stats['perimeter_to_vhs_ratio'] = float(np.mean(np.array(perimeters) / np.array(vhs_values)))
    
    return stats


def main(args):
    """Main function for prediction."""
    # Load configuration
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # Setup output directory
    output_dir = args.output if args.output else config.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_dir)
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    _, _, test_loader, _, _, test_dataset = create_dataloaders(
        base_path=config['data']['base_path'],
        batch_size=config['data']['batch_size'],
        augment=False  # No augmentation for test data
    )
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    
    # Create model architecture
    model = DogHeartViTWithAttention(
        pretrained=False,
        freeze_backbone=False,  # Doesn't matter for inference
        backbone=config['model']['backbone'],
        num_points=config['model']['num_points'],
        img_size=config['model']['img_size']
    )
    
    # Load weights
    model, _, _, _, _ = load_model(model, None, args.model, device)
    model.to(device)
    model.eval()
    
    # Make predictions
    logger.info("Making predictions on test set...")
    predictions = predict_test_set(
        model=model,
        test_loader=test_loader,
        output_path=os.path.join(output_dir, 'predictions.csv'),
        device=device
    )
    
    # Analyze predictions
    logger.info("Analyzing predictions...")
    stats = analyze_predictions(predictions)
    
    logger.info("Prediction statistics:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'prediction_stats.yaml')
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    logger.info(f"Prediction statistics saved to {stats_path}")
    
    # Visualize some predictions
    if args.visualize:
        logger.info("Visualizing predictions...")
        num_vis = min(args.num_visualizations, len(test_dataset))
        indices = np.random.choice(len(test_dataset), num_vis, replace=False)
        
        visualize_test_predictions(
            model=model,
            test_dataset=test_dataset,
            indices=indices,
            device=device
        )
    
    logger.info("Prediction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict dog heart VHS from images')
    parser.add_argument('--config', type=str, default='./config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--num-visualizations', type=int, default=5,
                        help='Number of visualizations to generate')
    
    args = parser.parse_args()
    
    main(args)