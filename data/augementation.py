"""
Data augmentation utilities for dog heart images with anatomical landmarks.
"""

import torch
import torchvision.transforms.functional as TF
import random
import math
import numpy as np


class AnatomicalAugmentation:
    """
    Anatomical augmentation for dog heart images with landmark points.
    
    This class applies various augmentations while preserving
    the anatomical integrity of the landmark points.
    
    Args:
        rotation_range (float): Maximum rotation angle in degrees (±)
        scale_range (float): Maximum scale factor variation (±)
        flip_prob (float): Probability of horizontal flip
        brightness_jitter (float): Maximum brightness adjustment (±)
        contrast_jitter (float): Maximum contrast adjustment (±)
    """
    def __init__(
        self,
        rotation_range=15.0,
        scale_range=0.1,
        flip_prob=0.5,
        brightness_jitter=0.1,
        contrast_jitter=0.1
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
    
    def __call__(self, image, points, vhs=None):
        """
        Apply augmentations to image and points.
        
        Args:
            image: PIL Image
            points: Numpy array of landmark points (N x 2)
            vhs: VHS value (optional, will be scaled if provided)
            
        Returns:
            tuple: (augmented_image, augmented_points, augmented_vhs)
        """
        width, height = image.size
        
        # Make a copy of points to avoid modifying original
        points = points.copy()
        
        # Track if VHS should be adjusted
        adjust_vhs = vhs is not None
        if adjust_vhs:
            vhs = float(vhs)
        
        # 1. Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            
            # Rotate image
            image = TF.rotate(image, angle, expand=False)
            
            # Calculate image center
            center_x, center_y = width / 2, height / 2
            
            # Rotate points around center
            angle_rad = math.radians(angle)
            cos_val = math.cos(angle_rad)
            sin_val = math.sin(angle_rad)
            
            for i in range(len(points)):
                x, y = points[i][0], points[i][1]
                
                # Translate to origin
                x -= center_x
                y -= center_y
                
                # Rotate
                new_x = x * cos_val - y * sin_val
                new_y = x * sin_val + y * cos_val
                
                # Translate back
                points[i][0] = new_x + center_x
                points[i][1] = new_y + center_y
        
        # 2. Random scaling
        if random.random() < 0.5:
            scale_factor = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            
            # Scale image
            new_width, new_height = int(width * scale_factor), int(height * scale_factor)
            image = TF.resize(image, (new_height, new_width))
            
            # Scale points
            for i in range(len(points)):
                points[i][0] *= scale_factor
                points[i][1] *= scale_factor
            
            # Scale VHS
            if adjust_vhs:
                vhs *= scale_factor
            
            # Resize back to original size
            image = TF.resize(image, (height, width))
            for i in range(len(points)):
                points[i][0] *= (width / new_width)
                points[i][1] *= (height / new_height)
        
        # 3. Random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            
            # Flip points horizontally
            for i in range(len(points)):
                points[i][0] = width - points[i][0]
        
        # 4. Color jitter
        if random.random() < 0.5:
            brightness_factor = random.uniform(1 - self.brightness_jitter, 1 + self.brightness_jitter)
            contrast_factor = random.uniform(1 - self.contrast_jitter, 1 + self.contrast_jitter)
            
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
        
        # Return results
        if adjust_vhs:
            return image, points, vhs
        else:
            return image, points


class RandomErasing:
    """
    Random erasing augmentation.
    
    This class randomly erases rectangular regions from the image
    to improve model robustness.
    
    Args:
        p (float): Probability of applying random erasing
        scale (tuple): Range of erasing area ratio (min, max)
        ratio (tuple): Range of aspect ratio (min, max)
        value (float or tuple): Value to fill erased area with
    """
    def __init__(self, p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        """
        Apply random erasing to image.
        
        Args:
            img: Tensor image (C x H x W)
            
        Returns:
            Tensor: Augmented image
        """
        if random.random() >= self.p:
            return img
        
        if not isinstance(img, torch.Tensor):
            raise TypeError("img should be a tensor image")
        
        # Get dimensions
        _, h, w = img.shape
        
        # Get erasing parameters
        erase_area = random.uniform(self.scale[0], self.scale[1]) * h * w
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        # Calculate erasing dimensions
        erase_h = int(round(math.sqrt(erase_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(erase_area / aspect_ratio)))
        
        # Ensure dimensions are within bounds
        if erase_h >= h or erase_w >= w:
            return img
        
        # Get random position
        top = random.randint(0, h - erase_h)
        left = random.randint(0, w - erase_w)
        
        # Create erasing mask
        mask = torch.ones_like(img)
        if isinstance(self.value, (int, float)):
            value = [self.value]
        else:
            value = self.value
        
        # Apply erasing
        img[:, top:top + erase_h, left:left + erase_w] = torch.tensor(value).view(-1, 1, 1)
        
        return img


def create_augmentation_transforms(augment_prob=0.5):
    """
    Create augmentation transforms for training images.
    
    Args:
        augment_prob (float): Probability of applying each augmentation
        
    Returns:
        callable: Augmentation function
    """
    anatomical_aug = AnatomicalAugmentation(
        rotation_range=15.0,
        scale_range=0.1,
        flip_prob=0.5
    )
    
    def augment_fn(image, points, vhs=None):
        if random.random() < augment_prob:
            return anatomical_aug(image, points, vhs)
        
        if vhs is not None:
            return image, points, vhs
        else:
            return image, points
    
    return augment_fn