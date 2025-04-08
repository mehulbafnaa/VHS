"""
Training script for dog heart VHS prediction model.
"""

import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dog_heart_vhs.data import create_dataloaders
from dog_heart_vhs.models.vit_regressor import DogHeartViTWithAttention
from dog_heart_vhs.models.loss import AnatomicalLoss
from dog_heart_vhs.utils.logger import setup_logger
from dog_heart_vhs.utils.metrics import compute_metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger=None):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger for printing updates
        
    Returns:
        dict: Dictionary of metrics
    """
    model.train()
    running_loss = 0.0
    running_metrics = {
        'vhs_mae': 0.0, 
        'points_mae': 0.0,
        'perimeter_error': 0.0
    }
    
    start_time = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for i, batch in enumerate(pbar):
        # Get data
        images = batch['image'].to(device)
        
        # Create targets dict
        targets = {}
        if 'points' in batch:
            targets['points'] = batch['points'].to(device)
        if 'perimeter' in batch:
            targets['perimeter'] = batch['perimeter'].to(device)
        if 'vhs' in batch:
            targets['vhs'] = batch['vhs'].unsqueeze(1).to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Check for NaN loss
        if torch.isnan(loss):
            if logger:
                logger.warning(f"NaN loss detected at batch {i}. Skipping.")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        
        # Calculate batch metrics
        batch_metrics = compute_metrics(outputs, targets)
        for k, v in batch_metrics.items():
            if k in running_metrics:
                running_metrics[k] += v
        
        # Update progress bar
        metrics_str = f"loss: {loss_dict['total_loss']:.4f}"
        if 'vhs_loss' in loss_dict:
            metrics_str += f", vhs: {loss_dict['vhs_loss']:.4f}"
        if 'points_loss' in loss_dict:
            metrics_str += f", pts: {loss_dict['points_loss']:.4f}"
        pbar.set_postfix_str(metrics_str)
    
    # Calculate average metrics
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    
    epoch_metrics = {
        'loss': epoch_loss,
        'time': time.time() - start_time
    }
    
    for k, v in running_metrics.items():
        epoch_metrics[k] = v / num_batches
    
    # Log epoch metrics
    if logger:
        logger.info(f"Train Epoch: {epoch} | Loss: {epoch_loss:.4f} | " +
                   f"VHS MAE: {epoch_metrics['vhs_mae']:.4f} | " +
                   f"Points MAE: {epoch_metrics['points_mae']:.4f} | " +
                   f"Time: {epoch_metrics['time']:.2f}s")
    
    return epoch_metrics


def validate(model, dataloader, criterion, device, epoch, logger=None):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger for printing updates
        
    Returns:
        dict: Dictionary of metrics
    """
    model.eval()
    running_loss = 0.0
    running_metrics = {
        'vhs_mae': 0.0, 
        'points_mae': 0.0,
        'perimeter_error': 0.0
    }
    
    start_time = time.time()
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")
    with torch.no_grad():
        for batch in pbar:
            # Get data
            images = batch['image'].to(device)
            
            # Create targets dict
            targets = {}
            if 'points' in batch:
                targets['points'] = batch['points'].to(device)
            if 'perimeter' in batch:
                targets['perimeter'] = batch['perimeter'].to(device)
            if 'vhs' in batch:
                targets['vhs'] = batch['vhs'].unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            
            # Calculate batch metrics
            batch_metrics = compute_metrics(outputs, targets)
            for k, v in batch_metrics.items():
                if k in running_metrics:
                    running_metrics[k] += v
            
            # Update progress bar
            metrics_str = f"loss: {loss_dict['total_loss']:.4f}"
            if 'vhs_loss' in loss_dict:
                metrics_str += f", vhs: {loss_dict['vhs_loss']:.4f}"
            if 'points_loss' in loss_dict:
                metrics_str += f", pts: {loss_dict['points_loss']:.4f}"
            pbar.set_postfix_str(metrics_str)
    
    # Calculate average metrics
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    
    epoch_metrics = {
        'loss': epoch_loss,
        'time': time.time() - start_time
    }
    
    for k, v in running_metrics.items():
        epoch_metrics[k] = v / num_batches
    
    # Log epoch metrics
    if logger:
        logger.info(f"Valid Epoch: {epoch} | Loss: {epoch_loss:.4f} | " +
                   f"VHS MAE: {epoch_metrics['vhs_mae']:.4f} | " +
                   f"Points MAE: {epoch_metrics['points_mae']:.4f} | " +
                   f"Time: {epoch_metrics['time']:.2f}s")
    
    return epoch_metrics


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        filename: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename, device):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filename: Path to checkpoint
        device: Device to load to
        
    Returns:
        tuple: (epoch, metrics)
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


def plot_metrics(train_metrics, val_metrics, save_path=None):
    """
    Plot training metrics.
    
    Args:
        train_metrics: Dictionary of train metrics for each epoch
        val_metrics: Dictionary of validation metrics for each epoch
        save_path: Path to save plot
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot loss
    axs[0].plot(train_metrics['loss'], label='Train')
    axs[0].plot(val_metrics['loss'], label='Validation')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Plot VHS MAE
    axs[1].plot(train_metrics['vhs_mae'], label='Train')
    axs[1].plot(val_metrics['vhs_mae'], label='Validation')
    axs[1].set_title('VHS Mean Absolute Error')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].legend()
    
    # Plot Points MAE
    axs[2].plot(train_metrics['points_mae'], label='Train')
    axs[2].plot(val_metrics['points_mae'], label='Validation')
    axs[2].set_title('Points Mean Absolute Error')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('MAE')
    axs[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def train_model(model, train_loader, val_loader, criterion=None, 
                epochs=30, initial_lr=1e-4, device='cuda', 
                output_dir='./output', logger=None, resume=None):
    """
    Train the full model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (defaults to AnatomicalLoss)
        epochs: Number of epochs to train
        initial_lr: Initial learning rate
        device: Device to train on
        output_dir: Directory to save outputs
        logger: Logger for printing updates
        resume: Path to checkpoint to resume from
        
    Returns:
        tuple: (model, history)
    """
    model = model.to(device)
    
    # Create default criterion if not provided
    if criterion is None:
        criterion = AnatomicalLoss(
            points_weight=1.0,
            vhs_weight=10.0,
            perimeter_weight=0.0  # Start with perimeter loss disabled
        )
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=initial_lr,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics storage
    history = {
        'train': {
            'loss': [], 'vhs_mae': [], 'points_mae': [], 'perimeter_error': []
        },
        'val': {
            'loss': [], 'vhs_mae': [], 'points_mae': [], 'perimeter_error': []
        }
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    last_model_path = os.path.join(output_dir, 'last_model.pth')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume and os.path.exists(resume):
        if logger:
            logger.info(f"Resuming from checkpoint: {resume}")
        start_epoch, resume_metrics = load_checkpoint(model, optimizer, resume, device)
        start_epoch += 1  # Start from the next epoch
        
        # Restore history if available
        if 'history' in resume_metrics:
            history = resume_metrics['history']
        
        if logger:
            logger.info(f"Resumed from epoch {start_epoch}")
    
    # Print initial data statistics
    if logger:
        logger.info("\nChecking data scales:")
        with torch.no_grad():
            for batch in train_loader:
                images = batch['image']
                points = batch['points'] if 'points' in batch else None
                vhs = batch['vhs'] if 'vhs' in batch else None
                perimeter = batch['perimeter'] if 'perimeter' in batch else None
                
                logger.info(f"Image shape: {images.shape}, range: [{images.min().item():.4f}, {images.max().item():.4f}]")
                if points is not None:
                    logger.info(f"Points shape: {points.shape}, range: [{points.min().item():.4f}, {points.max().item():.4f}]")
                if vhs is not None:
                    logger.info(f"VHS shape: {vhs.shape}, range: [{vhs.min().item():.4f}, {vhs.max().item():.4f}]")
                if perimeter is not None:
                    logger.info(f"Perimeter shape: {perimeter.shape}, range: [{perimeter.min().item():.4f}, {perimeter.max().item():.4f}]")
                    if vhs is not None:
                        ratio = perimeter / vhs
                        logger.info(f"Perimeter/VHS ratio range: [{ratio.min().item():.4f}, {ratio.max().item():.4f}], mean: {ratio.mean().item():.4f}")
                break
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        # Training phase
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger
        )
        
        # Validation phase
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            logger=logger
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_metrics['loss'])
        
        # Save metrics
        for k, v in train_metrics.items():
            if k in history['train']:
                history['train'][k].append(v)
        
        for k, v in val_metrics.items():
            if k in history['val']:
                history['val'][k].append(v)
        
        # Save last checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'history': history},
            filename=last_model_path
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'history': history},
                filename=best_model_path
            )
            if logger:
                logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Plot metrics
        plot_metrics(
            train_metrics=history['train'],
            val_metrics=history['val'],
            save_path=os.path.join(output_dir, 'metrics.png')
        )
        
        # Enable perimeter ratio loss after 10 epochs if training is stable
        if epoch == 9 and hasattr(criterion, 'perimeter_weight') and criterion.perimeter_weight == 0:
            criterion.perimeter_weight = 0.1
            if logger:
                logger.info("Enabling perimeter ratio loss with weight 0.1")
    
    # Load best model
    if os.path.exists(best_model_path):
        if logger:
            logger.info(f"Loading best model from {best_model_path}")
        _, _ = load_checkpoint(model, optimizer, best_model_path, device)
    
    return model, history


def main(args):
    """Main function for training."""
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
    logger.info(f"Configuration: {config}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        base_path=config['data']['base_path'],
        batch_size=config['data']['batch_size'],
        augment=config['data']['augment']
    )
    
    # Create model
    logger.info("Creating model...")
    model = DogHeartViTWithAttention(
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        backbone=config['model']['backbone'],
        num_points=config['model']['num_points'],
        img_size=config['model']['img_size']
    )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    logger.info("Creating loss function...")
    criterion = AnatomicalLoss(
        points_weight=config['loss']['points_weight'],
        vhs_weight=config['loss']['vhs_weight'],
        perimeter_weight=config['loss']['perimeter_weight'],
        shape_weight=config['loss']['shape_weight']
    )
    
    # Train model
    logger.info("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=config['training']['epochs'],
        initial_lr=config['training']['learning_rate'],
        device=device,
        output_dir=output_dir,
        logger=logger,
        resume=args.resume
    )
    
    logger.info("Training complete!")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train dog heart VHS prediction model')
    parser.add_argument('--config', type=str, default='./config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    main(args)