"""
Evaluation metrics for dog heart VHS prediction.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(outputs, targets):
    """
    Compute evaluation metrics for predictions.
    
    Args:
        outputs (dict): Model outputs
        targets (dict): Target values
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # VHS Mean Absolute Error
    if 'vhs' in outputs and 'vhs' in targets:
        vhs_mae = torch.abs(outputs['vhs'] - targets['vhs']).mean().item()
        metrics['vhs_mae'] = vhs_mae
    
    # Points Mean Absolute Error
    if 'points' in outputs and 'points' in targets:
        points_mae = torch.abs(outputs['points'] - targets['points']).mean().item()
        metrics['points_mae'] = points_mae
    
    # Perimeter Error
    if 'perimeter' in outputs and 'perimeter' in targets:
        perimeter_error = torch.abs(outputs['perimeter'] - targets['perimeter']).mean().item()
        metrics['perimeter_error'] = perimeter_error
        
        # Perimeter-to-VHS Ratio Error
        if 'vhs' in outputs and 'vhs' in targets:
            pred_ratio = outputs['perimeter'] / outputs['vhs'].squeeze()
            target_ratio = targets['perimeter'] / targets['vhs'].squeeze()
            ratio_error = torch.abs(pred_ratio - target_ratio).mean().item()
            metrics['ratio_error'] = ratio_error
    
    return metrics


def calculate_vhs_from_points(points):
    """
    Calculate VHS from predicted points using the perimeter-to-VHS ratio.
    
    Args:
        points (tensor): Predicted points of shape [batch_size, num_points, 2]
        
    Returns:
        tensor: Estimated VHS values of shape [batch_size]
    """
    batch_size, num_points, _ = points.shape
    
    # Calculate perimeter
    perimeter = torch.zeros(batch_size, device=points.device)
    for i in range(num_points):
        next_i = (i + 1) % num_points
        p1 = points[:, i, :]
        p2 = points[:, next_i, :]
        dist = torch.sqrt(torch.sum((p2 - p1) ** 2, dim=1))
        perimeter += dist
    
    # Use average perimeter-to-VHS ratio from training data
    # This is a placeholder value - replace with actual value from your dataset
    avg_ratio = 114.1
    
    # Calculate VHS
    vhs = perimeter / avg_ratio
    
    return vhs


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix for classification tasks.
    
    Args:
        y_true (tensor or array): Ground truth labels
        y_pred (tensor or array): Predicted labels
        num_classes (int, optional): Number of classes
        
    Returns:
        array: Confusion matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate confusion matrix
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm


def regression_metrics(y_true, y_pred):
    """
    Compute standard regression metrics.
    
    Args:
        y_true (tensor or array): Ground truth values
        y_pred (tensor or array): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Ensure arrays are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Compute additional metrics
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'max_error': max_error
    }


def point_distance_error(pred_points, true_points):
    """
    Calculate the distance error for each predicted point.
    
    Args:
        pred_points (tensor): Predicted points of shape [batch_size, num_points, 2]
        true_points (tensor): True points of shape [batch_size, num_points, 2]
        
    Returns:
        dict: Dictionary of metrics
    """
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.cpu().numpy()
    if isinstance(true_points, torch.Tensor):
        true_points = true_points.cpu().numpy()
    
    batch_size, num_points, _ = pred_points.shape
    
    # Calculate Euclidean distance for each point
    point_distances = np.sqrt(np.sum((pred_points - true_points)**2, axis=2))
    
    # Calculate metrics for each point
    point_metrics = {}
    for i in range(num_points):
        point_metrics[f'point_{i+1}_distance'] = np.mean(point_distances[:, i])
    
    # Calculate overall metrics
    point_metrics['mean_distance'] = np.mean(point_distances)
    point_metrics['max_distance'] = np.max(point_distances)
    point_metrics['min_distance'] = np.min(point_distances)
    point_metrics['std_distance'] = np.std(point_distances)
    
    return point_metrics


def anatomical_consistency_score(points):
    """
    Calculate a score for anatomical consistency of predicted points.
    
    This function checks if the predicted points form a plausible anatomical shape
    by analyzing their relative positions and angles.
    
    Args:
        points (tensor or array): Predicted points of shape [batch_size, num_points, 2]
        
    Returns:
        float: Consistency score between 0 and 1 (higher is better)
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    batch_size, num_points, _ = points.shape
    
    # Calculate centroid for each sample
    centroids = np.mean(points, axis=1)
    
    # Calculate consistency scores
    scores = np.zeros(batch_size)
    
    for i in range(batch_size):
        # Calculate points relative to centroid
        centered_points = points[i] - centroids[i]
        
        # Calculate angles between consecutive points
        angles = []
        for j in range(num_points):
            p1 = centered_points[j]
            p2 = centered_points[(j + 1) % num_points]
            
            # Calculate angle
            angle = np.arctan2(p2[1], p2[0]) - np.arctan2(p1[1], p1[0])
            # Normalize to [-pi, pi]
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            angles.append(angle)
        
        # Calculate angle consistency (sum of angles for a simple polygon should be (n-2)*pi)
        expected_sum = (num_points - 2) * np.pi
        actual_sum = np.abs(np.sum(angles))
        angle_score = 1.0 - min(1.0, np.abs(actual_sum - expected_sum) / np.pi)
        
        # Calculate convexity score (proportion of angles with same sign)
        sign_changes = np.sum(np.diff(np.signbit(angles)) != 0)
        convexity_score = 1.0 - min(1.0, sign_changes / num_points)
        
        # Combine scores (can be weighted differently if needed)
        scores[i] = 0.5 * angle_score + 0.5 * convexity_score
    
    return np.mean(scores)


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_vhs_true = []
    all_vhs_pred = []
    all_points_true = []
    all_points_pred = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            images = batch['image'].to(device)
            
            # Create targets dict
            targets = {}
            if 'points' in batch:
                targets['points'] = batch['points'].to(device)
                all_points_true.append(batch['points'].cpu().numpy())
            if 'perimeter' in batch:
                targets['perimeter'] = batch['perimeter'].to(device)
            if 'vhs' in batch:
                targets['vhs'] = batch['vhs'].unsqueeze(1).to(device)
                all_vhs_true.append(batch['vhs'].unsqueeze(1).cpu().numpy())
            
            # Forward pass
            outputs = model(images)
            
            # Collect predictions
            if 'vhs' in outputs:
                all_vhs_pred.append(outputs['vhs'].cpu().numpy())
            if 'points' in outputs:
                all_points_pred.append(outputs['points'].cpu().numpy())
    
    # Combine predictions
    results = {}
    
    if all_vhs_true and all_vhs_pred:
        vhs_true = np.concatenate(all_vhs_true)
        vhs_pred = np.concatenate(all_vhs_pred)
        vhs_metrics = regression_metrics(vhs_true, vhs_pred)
        
        # Add VHS metrics to results
        for k, v in vhs_metrics.items():
            results[f'vhs_{k}'] = v
    
    if all_points_true and all_points_pred:
        points_true = np.concatenate(all_points_true)
        points_pred = np.concatenate(all_points_pred)
        
        # Calculate point-wise metrics
        point_metrics = point_distance_error(points_pred, points_true)
        
        # Add point metrics to results
        for k, v in point_metrics.items():
            results[k] = v
        
        # Calculate anatomical consistency
        consistency_score = anatomical_consistency_score(points_pred)
        results['anatomical_consistency'] = consistency_score
    
    return results