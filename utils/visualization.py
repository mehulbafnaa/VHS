"""
Visualization utilities for dog heart VHS prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from PIL import Image


def visualize_points(image, points, vhs=None, title=None, ax=None, show=True):
    """
    Visualize landmark points on an image.
    
    Args:
        image: PIL Image or numpy array
        points: Array of shape [num_points, 2]
        vhs (float, optional): VHS value to display
        title (str, optional): Plot title
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        show (bool): Whether to show the plot
        
    Returns:
        matplotlib.axes.Axes: Axes with the plot
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create new axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display image
    ax.imshow(image)
    
    # Plot the six points
    for i, point in enumerate(points):
        ax.scatter(point[0], point[1], c='r', s=50)
        ax.text(point[0] + 5, point[1] + 5, str(i+1), color='white', 
                bbox=dict(facecolor='red', alpha=0.7))
    
    # Plot connections between points to form the polygon
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        ax.plot([points[i][0], points[next_i][0]], 
               [points[i][1], points[next_i][1]], 'r-')
    
    # Plot centroid
    centroid_x = sum(p[0] for p in points) / len(points)
    centroid_y = sum(p[1] for p in points) / len(points)
    ax.scatter(centroid_x, centroid_y, c='blue', s=100, marker='x')
    
    # Calculate perimeter
    perimeter = 0
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        p1 = points[i]
        p2 = points[next_i]
        dist = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
        perimeter += dist
    
    # Set title
    if title is None:
        if vhs is not None:
            title = f"VHS: {vhs:.2f}, Perimeter: {perimeter:.2f}, Ratio: {perimeter/vhs:.2f}"
        else:
            title = f"Perimeter: {perimeter:.2f}"
    
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def visualize_comparison(image, true_points, pred_points, true_vhs=None, pred_vhs=None, ax=None, show=True):
    """
    Visualize comparison between true and predicted landmark points.
    
    Args:
        image: PIL Image or numpy array
        true_points: Array of true landmark points
        pred_points: Array of predicted landmark points
        true_vhs (float, optional): True VHS value
        pred_vhs (float, optional): Predicted VHS value
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        show (bool): Whether to show the plot
        
    Returns:
        matplotlib.axes.Axes: Axes with the plot
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create new axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display image
    ax.imshow(image)
    
    # Plot true points in green
    for i, point in enumerate(true_points):
        ax.scatter(point[0], point[1], c='g', s=50, alpha=0.7, label='True' if i == 0 else None)
        ax.text(point[0] + 5, point[1] + 5, str(i+1), color='white', 
                bbox=dict(facecolor='green', alpha=0.5))
    
    # Connect true points
    true_poly = np.array(true_points)
    ax.add_patch(Polygon(true_poly, fill=False, edgecolor='g', linewidth=2, alpha=0.7))
    
    # Plot predicted points in red
    for i, point in enumerate(pred_points):
        ax.scatter(point[0], point[1], c='r', s=50, alpha=0.7, label='Predicted' if i == 0 else None)
        ax.text(point[0] + 5, point[1] + 5, str(i+1), color='white', 
                bbox=dict(facecolor='red', alpha=0.5))
    
    # Connect predicted points
    pred_poly = np.array(pred_points)
    ax.add_patch(Polygon(pred_poly, fill=False, edgecolor='r', linewidth=2, alpha=0.7))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set title
    title = ""
    if true_vhs is not None and pred_vhs is not None:
        title = f"True VHS: {true_vhs:.2f}, Predicted VHS: {pred_vhs:.2f}, Error: {abs(true_vhs - pred_vhs):.2f}"
    
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def visualize_batch(batch_images, batch_points=None, batch_vhs=None, max_samples=16, figsize=(15, 15)):
    """
    Visualize a batch of images with points and VHS values.
    
    Args:
        batch_images: Tensor of images [batch_size, channels, height, width]
        batch_points (optional): Tensor of points [batch_size, num_points, 2]
        batch_vhs (optional): Tensor of VHS values [batch_size]
        max_samples (int): Maximum number of samples to show
        figsize (tuple): Figure size
    """
    # Convert tensors to numpy
    if isinstance(batch_images, torch.Tensor):
        # Convert from [B, C, H, W] to [B, H, W, C]
        images = batch_images.permute(0, 2, 3, 1).cpu().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        images = images * std + mean
        
        # Clip to [0, 1]
        images = np.clip(images, 0, 1)
    else:
        images = batch_images
    
    if batch_points is not None and isinstance(batch_points, torch.Tensor):
        points = batch_points.cpu().numpy()
    else:
        points = batch_points
    
    if batch_vhs is not None and isinstance(batch_vhs, torch.Tensor):
        vhs = batch_vhs.cpu().numpy()
    else:
        vhs = batch_vhs
    
    # Limit number of samples
    num_samples = min(len(images), max_samples)
    images = images[:num_samples]
    if points is not None:
        points = points[:num_samples]
    if vhs is not None:
        vhs = vhs[:num_samples]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each sample
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        
        # Add points if available
        if points is not None:
            for j, point in enumerate(points[i]):
                ax.scatter(point[0], point[1], c='r', s=20)
                # Add point number
                ax.text(point[0] + 2, point[1] + 2, str(j+1), color='white', 
                        fontsize=8, bbox=dict(facecolor='red', alpha=0.7))
            
            # Connect points
            for j in range(len(points[i])):
                next_j = (j + 1) % len(points[i])
                ax.plot([points[i][j][0], points[i][next_j][0]], 
                       [points[i][j][1], points[i][next_j][1]], 'r-', linewidth=1)
        
        # Add VHS value if available
        if vhs is not None:
            ax.set_title(f"VHS: {vhs[i]:.2f}")
        
        ax.axis('off')
    
    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_attention_weights(image, attention_weights, points=None, vhs=None, figsize=(15, 10)):
    """
    Visualize attention weights overlaid on an image.
    
    Args:
        image: PIL Image or numpy array
        attention_weights: Attention weights [num_heads, height, width]
        points (optional): Landmark points to overlay
        vhs (optional): VHS value to display
        figsize (tuple): Figure size
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert attention weights to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Number of attention heads
    num_heads = attention_weights.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, num_heads + 1, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    
    # Add points and VHS if available
    if points is not None:
        for i, point in enumerate(points):
            axes[0].scatter(point[0], point[1], c='r', s=30)
            axes[0].text(point[0] + 5, point[1] + 5, str(i+1), color='white', 
                    fontsize=10, bbox=dict(facecolor='red', alpha=0.7))
        
        # Connect points
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            axes[0].plot([points[i][0], points[next_i][0]], 
                        [points[i][1], points[next_i][1]], 'r-', linewidth=1.5)
    
    if vhs is not None:
        title = axes[0].get_title()
        axes[0].set_title(f"{title}\nVHS: {vhs:.2f}")
    
    axes[0].axis('off')
    
    # Plot attention weights for each head
    for h in range(num_heads):
        ax = axes[h + 1]
        
        # Resize attention weights to match image size if needed
        if attention_weights[h].shape != image.shape[:2]:
            from skimage.transform import resize
            attn = resize(attention_weights[h], image.shape[:2], order=1, mode='constant')
        else:
            attn = attention_weights[h]
        
        # Display image
        ax.imshow(image)
        
        # Overlay attention weights
        attn_masked = np.ma.masked_where(attn < 0.1, attn)  # Mask low attention values
        ax.imshow(attn_masked, cmap='jet', alpha=0.5)
        
        ax.set_title(f"Attention Head {h+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_test_predictions(model, test_dataset, indices=None, num_samples=5, device='cuda'):
    """
    Visualize model predictions on test images.
    
    Args:
        model: PyTorch model
        test_dataset: Test dataset
        indices (list, optional): Specific indices to visualize
        num_samples (int): Number of random samples to visualize if indices not provided
        device (str): Device to run model on
    """
    model.eval()
    
    # Select random indices if not provided
    if indices is None:
        indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 6*len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    # Generate predictions for each sample
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            sample = test_dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            
            # Get predictions
            outputs = model(image)
            
            # Get predicted points and VHS
            if 'points' in outputs:
                pred_points = outputs['points'][0].cpu().numpy()
            else:
                pred_points = None
            
            if 'vhs' in outputs:
                pred_vhs = outputs['vhs'][0].item()
            else:
                pred_vhs = None
            
            # Convert image for display
            if isinstance(sample['image'], torch.Tensor):
                # Denormalize
                img = sample['image'].permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)
            else:
                img = sample['image']
            
            # Display image with predictions
            ax = axes[i]
            ax.imshow(img)
            
            if pred_points is not None:
                # Draw points
                for j, point in enumerate(pred_points):
                    ax.scatter(point[0], point[1], c='r', s=50)
                    ax.text(point[0] + 5, point[1] + 5, str(j+1), color='white', 
                            fontsize=10, bbox=dict(facecolor='red', alpha=0.7))
                
                # Connect points
                for j in range(len(pred_points)):
                    next_j = (j + 1) % len(pred_points)
                    ax.plot([pred_points[j][0], pred_points[next_j][0]], 
                           [pred_points[j][1], pred_points[next_j][1]], 'r-', linewidth=1.5)
            
            # Add title with filename and predicted VHS
            title = f"File: {sample['filename']}"
            if pred_vhs is not None:
                title += f"\nPredicted VHS: {pred_vhs:.2f}"
            
            ax.set_title(title)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history with 'train' and 'val' dictionaries
        save_path (str, optional): Path to save the plot
    """
    # Extract metrics
    train_metrics = history['train']
    val_metrics = history['val']
    
    # Get all metrics
    all_metrics = set()
    for metric in train_metrics.keys():
        all_metrics.add(metric)
    for metric in val_metrics.keys():
        all_metrics.add(metric)
    
    # Remove 'time' metric
    if 'time' in all_metrics:
        all_metrics.remove('time')
    
    # Calculate number of rows and columns for subplot grid
    n_metrics = len(all_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Convert to list of axes if there's only one row
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(sorted(all_metrics)):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row][col] if n_rows > 1 and n_cols > 1 else axes[i]
        
        # Plot train metric if available
        if metric in train_metrics and len(train_metrics[metric]) > 0:
            ax.plot(train_metrics[metric], label='Train', marker='o', markersize=3)
        
        # Plot val metric if available
        if metric in val_metrics and len(val_metrics[metric]) > 0:
            ax.plot(val_metrics[metric], label='Validation', marker='s', markersize=3)
        
        # Set title and labels
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused axes
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1 and n_cols > 1:
            axes[row][col].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()