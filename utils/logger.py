"""
Logging utilities for dog heart VHS prediction.
"""

import os
import logging
import sys
import time
from datetime import datetime


def setup_logger(output_dir=None, name="dog_heart_vhs"):
    """
    Set up logger for the project.
    
    Args:
        output_dir (str, optional): Directory to save log file
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    # Prevent logger from duplicating messages when used in multiple modules
    logger.propagate = False
    
    return logger


class Logger:
    """
    Logger class for training progress.
    
    This class provides methods for logging training progress to console
    and optionally to TensorBoard.
    
    Args:
        log_dir (str, optional): Directory to save logs
        use_tensorboard (bool): Whether to use TensorBoard for logging
    """
    def __init__(self, log_dir=None, use_tensorboard=False):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        # Create logger
        self.logger = setup_logger(log_dir)
        
        # Setup TensorBoard if requested
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                self.logger.info(f"TensorBoard logging enabled at {log_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
                self.writer = None
        else:
            self.writer = None
    
    def log_metrics(self, metrics, step, prefix=""):
        """
        Log metrics to console and TensorBoard.
        
        Args:
            metrics (dict): Dictionary of metrics
            step (int): Training step or epoch
            prefix (str): Prefix for metric names (e.g., 'train/', 'val/')
        """
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metrics_str}")
        
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}{k}", v, step)
    
    def log_hyperparameters(self, hparams):
        """
        Log hyperparameters.
        
        Args:
            hparams (dict): Dictionary of hyperparameters
        """
        # Log to console
        self.logger.info("Hyperparameters:")
        for k, v in hparams.items():
            self.logger.info(f"  {k}: {v}")
        
        # Log to TensorBoard
        if self.use_tensorboard and self.writer is not None:
            from torch.utils.tensorboard.summary import hparams
            self.writer.add_hparams(hparams, {})
    
    def log_images(self, images, step, tag="images"):
        """
        Log images to TensorBoard.
        
        Args:
            images (tensor): Batch of images [B, C, H, W]
            step (int): Training step or epoch
            tag (str): Tag for the images
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_images(tag, images, step)
    
    def log_model_graph(self, model, input_size=(1, 3, 224, 224)):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            input_size (tuple): Input tensor size
        """
        if self.use_tensorboard and self.writer is not None:
            import torch
            device = next(model.parameters()).device
            x = torch.zeros(input_size, device=device)
            self.writer.add_graph(model, x)
    
    def close(self):
        """Close the logger and TensorBoard writer."""
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()


class TqdmToLogger:
    """
    Output stream for tqdm which will output to logger.
    
    Args:
        logger: Logger instance
        level (int): Logging level
    """
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ""
    
    def write(self, buf):
        self.buffer = buf.strip("\r\n\t ")
    
    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer)
            self.buffer = ""


class Timer:
    """
    Timer class for measuring execution time.
    
    Example:
        ```
        timer = Timer()
        # Do some work
        elapsed = timer.elapsed()
        print(f"Elapsed time: {elapsed:.2f} seconds")
        ```
    """
    def __init__(self):
        self.start_time = time.time()
    
    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()
    
    def elapsed(self):
        """
        Get elapsed time in seconds.
        
        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.start_time