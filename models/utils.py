"""
Utility functions for models.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import random


def set_seed(seed):
    """
    Set seed for reproducibility.
    
    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model):
    """
    Freeze backbone parameters in a model.
    
    Args:
        model: PyTorch model with a 'backbone' attribute
        
    Returns:
        model: Model with frozen backbone
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    return model


def unfreeze_backbone(model):
    """
    Unfreeze backbone parameters in a model.
    
    Args:
        model: PyTorch model with a 'backbone' attribute
        
    Returns:
        model: Model with unfrozen backbone
    """
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    return model


def save_model(model, optimizer, epoch, metrics, config, path):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        metrics (dict): Dictionary of metrics
        config (dict): Dictionary of configuration parameters
        path (str): Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, path)


def load_model(model, optimizer, path, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path (str): Path to checkpoint
        device: Device to load checkpoint to
        
    Returns:
        tuple: (model, optimizer, epoch, metrics, config)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    config = checkpoint.get('config', {})
    
    return model, optimizer, epoch, metrics, config


def create_model_from_checkpoint(model_class, checkpoint_path, device):
    """
    Create a model from a checkpoint.
    
    Args:
        model_class: PyTorch model class
        checkpoint_path (str): Path to checkpoint
        device: Device to load checkpoint to
        
    Returns:
        model: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model using config
    model = model_class(**config.get('model', {}))
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model


def initialize_weights(model):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        
    Returns:
        model: Model with initialized weights
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model


def create_optimizer(model, optimizer_name, lr, weight_decay=0):
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        optimizer_name (str): Name of optimizer ('adam', 'adamw', 'sgd')
        lr (float): Learning rate
        weight_decay (float): Weight decay
        
    Returns:
        optimizer: PyTorch optimizer
    """
    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_name, **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name (str): Name of scheduler
        **kwargs: Additional arguments for scheduler
        
    Returns:
        scheduler: PyTorch scheduler
    """
    if scheduler_name == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'plateau':
        patience = kwargs.get('patience', 5)
        factor = kwargs.get('factor', 0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    elif scheduler_name == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Mixup data augmentation.
    
    Args:
        x (tensor): Batch of images [batch_size, channels, height, width]
        y (tensor): Batch of targets
        alpha (float): Mixup parameter
        device (str): Device to use
        
    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    CutMix data augmentation.
    
    Args:
        x (tensor): Batch of images [batch_size, channels, height, width]
        y (tensor): Batch of targets
        alpha (float): CutMix parameter
        device (str): Device to use
        
    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    # Get dimensions
    _, _, h, w = x.shape
    
    # Calculate cutmix box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    
    # Get random box
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def convert_model_to_onnx(model, save_path, input_size=(1, 3, 224, 224), device='cpu'):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        save_path (str): Path to save ONNX model
        input_size (tuple): Input tensor size
        device (str): Device to use
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )