"""
Dataloaders for dog heart VHS prediction.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import random
import math


class DogHeartPointsDataset(Dataset):
    """
    Dataset for dog heart images with six_points landmarks.
    
    This dataset loads .mat files containing VHS values and landmark points,
    and matches them with corresponding image files.
    
    Args:
        label_dir (str): Directory containing .mat files with VHS and six_points
        image_dir (str): Directory containing image files
        transform (callable, optional): Standard image transforms (applied after augmentation)
        augment (bool): Whether to use augmentation based on six_points
        augment_prob (float): Probability of applying augmentation
    """
    def __init__(self, label_dir, image_dir, transform=None, augment=True, augment_prob=0.5):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment
        self.augment_prob = augment_prob
        
        # For storing samples with both images and six_points
        self.samples = []  
        self.vhs_values = []
        
        # Get all available image files for quick lookup
        self.available_images = set(os.listdir(image_dir))
        print(f"Found {len(self.available_images)} images in image directory")
        
        # Process all .mat files in the label directory
        mat_files = [f for f in os.listdir(label_dir) if f.endswith('.mat')]
        if not mat_files:
            raise ValueError(f"No .mat files found in label directory: {label_dir}")

        print(f"Found {len(mat_files)} .mat files in label directory")
        
        # Process each .mat file
        for mat_file in mat_files:
            mat_path = os.path.join(label_dir, mat_file)
            try:
                # Load the .mat file
                mat_data = sio.loadmat(mat_path)
                
                # Check if file has both VHS and six_points
                if 'VHS' not in mat_data or 'six_points' not in mat_data:
                    continue
                    
                # Extract VHS value
                vhs_data = mat_data['VHS']
                vhs_value = float(vhs_data.item()) if isinstance(vhs_data, np.ndarray) else float(vhs_data)
                
                # Extract six_points
                six_points = mat_data['six_points'].astype(np.float32)
                
                # Try to find matching image
                base_filename = os.path.splitext(mat_file)[0]
                image_name = f"{base_filename}.png"
                
                # Try additional naming patterns if needed
                if image_name not in self.available_images:
                    # Extract the ID part (assuming format like "64_10.2_1.mat" -> "64.png")
                    id_part = base_filename.split('_')[0]
                    image_name = f"{id_part}.png"
                
                # Try with zero padding to 4 digits
                if image_name not in self.available_images:
                    try:
                        id_num = int(id_part)
                        image_name = f"{id_num:04d}.png"  # Zero-padded to 4 digits
                    except ValueError:
                        # If id_part is not a valid integer, skip this attempt
                        pass
                
                # If image found, add to samples
                if image_name in self.available_images:
                    image_path = os.path.join(image_dir, image_name)
                    self.samples.append({
                        'image_path': image_path,
                        'six_points': six_points,
                        'vhs_value': vhs_value
                    })
                    self.vhs_values.append(vhs_value)
                else:
                    print(f"Warning: No matching image found for {mat_file}")
            
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                
        print(f"Successfully loaded {len(self.samples)} samples with valid images, points, and VHS values")
        
        if self.vhs_values:
            min_vhs = min(self.vhs_values)
            max_vhs = max(self.vhs_values)
            mean_vhs = sum(self.vhs_values) / len(self.vhs_values)
            print(f"VHS values - Min: {min_vhs:.2f}, Max: {max_vhs:.2f}, Mean: {mean_vhs:.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Get points and VHS
        points = sample['six_points'].copy()
        vhs = sample['vhs_value']
        
        # Get image dimensions
        width, height = image.size
        
        # Apply augmentation using six_points
        if self.augment and random.random() < self.augment_prob:
            # 1. Random rotation (±15 degrees)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                
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
            
            # 2. Random scaling (±10%)
            if random.random() < 0.5:
                scale_factor = random.uniform(0.9, 1.1)
                
                # Scale image
                new_width, new_height = int(width * scale_factor), int(height * scale_factor)
                image = TF.resize(image, (new_height, new_width))
                
                # Scale points
                for i in range(len(points)):
                    points[i][0] *= scale_factor
                    points[i][1] *= scale_factor
                
                # Scale VHS (assuming proportional relationship)
                vhs *= scale_factor
                
                # Resize back to original size
                image = TF.resize(image, (height, width))
                for i in range(len(points)):
                    points[i][0] *= (width / new_width)
                    points[i][1] *= (height / new_height)
            
            # 3. Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                
                # Flip points horizontally
                for i in range(len(points)):
                    points[i][0] = width - points[i][0]
        
        # Apply standard transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert points to tensor
        points_tensor = torch.tensor(points, dtype=torch.float32)
        
        # Calculate perimeter (used for the ratio to VHS)
        perimeter = 0
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            p1 = points[i]
            p2 = points[next_i]
            dist = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
            perimeter += dist
        
        # Return image, points, perimeter and VHS
        return {
            'image': image,
            'points': points_tensor,
            'perimeter': torch.tensor(perimeter, dtype=torch.float32),
            'vhs': torch.tensor(vhs, dtype=torch.float32)
        }
    
    def visualize_sample(self, idx):
        """
        Visualize a sample with its six points and VHS value.
        
        Args:
            idx (int): Index of sample to visualize
            
        Returns:
            tuple: (image, points, vhs)
        """
        sample = self.samples[idx]
        
        # Load image
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        points = sample['six_points']
        vhs = sample['vhs_value']
        
        # Calculate perimeter
        perimeter = 0
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            p1 = points[i]
            p2 = points[next_i]
            dist = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
            perimeter += dist
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        
        # Plot the six points
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c='r', s=50)
            plt.text(point[0] + 5, point[1] + 5, str(i+1), color='white', 
                    bbox=dict(facecolor='red', alpha=0.7))
        
        # Plot connections between points to form the polygon
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            plt.plot([points[i][0], points[next_i][0]], 
                    [points[i][1], points[next_i][1]], 'r-')
        
        # Plot centroid
        centroid_x = sum(p[0] for p in points) / len(points)
        centroid_y = sum(p[1] for p in points) / len(points)
        plt.scatter(centroid_x, centroid_y, c='blue', s=100, marker='x')
        
        plt.title(f"VHS: {vhs:.2f}, Perimeter: {perimeter:.2f}, Ratio: {perimeter/vhs:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return image, points, vhs


class DogHeartTestDataset(Dataset):
    """
    Dataset for test images (without labels/points).
    
    Args:
        test_dir (str): Directory containing test images
        transform (callable, optional): Image transforms
    """
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = []
        self.filenames = []

        # Get all image files from the test directory
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith(('.png')):  # Only looking for PNG files
                    self.image_paths.append(os.path.join(test_dir, file))
                    self.filenames.append(file)
            print(f"Found {len(self.image_paths)} test images in {test_dir}")
        else:
            print(f"Warning: Test directory {test_dir} does not exist!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = self.filenames[idx]

        # Open and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'filename': filename
        }


def create_dataloaders(base_path, batch_size=16, augment=True):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        base_path (str): Base directory containing data folders
        batch_size (int): Batch size for data loaders
        augment (bool): Whether to use augmentation for training
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    # Paths
    train_label_dir = os.path.join(base_path, "Train/Labels")
    train_image_dir = os.path.join(base_path, "Train/Images")
    val_label_dir = os.path.join(base_path, "Valid/Labels")
    val_image_dir = os.path.join(base_path, "Valid/Images")
    test_dir = os.path.join(base_path, "Test_Images/Images")
    
    print(f"Training label path: {train_label_dir}")
    print(f"Training image path: {train_image_dir}")
    print(f"Validation label path: {val_label_dir}")
    print(f"Validation image path: {val_image_dir}")
    print(f"Test path: {test_dir}")
    
    # Check if directories exist
    print("\nChecking directories:")
    for directory in [train_label_dir, train_image_dir, val_label_dir, val_image_dir, test_dir]:
        if os.path.exists(directory):
            print(f"Directory {directory} exists.")
            print(f"Contents: {os.listdir(directory)[:5]}...") # Show only first 5 items
        else:
            print(f"Directory {directory} does not exist!")
    
    # Standard transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    try:
        train_dataset = DogHeartPointsDataset(
            label_dir=train_label_dir,
            image_dir=train_image_dir,
            transform=train_transforms,
            augment=augment
        )
        
        val_dataset = DogHeartPointsDataset(
            label_dir=val_label_dir,
            image_dir=val_image_dir,
            transform=val_test_transforms,
            augment=False  # No augmentation for validation
        )
        
        test_dataset = DogHeartTestDataset(
            test_dir=test_dir,
            transform=val_test_transforms
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Print dataset sizes
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        
        # Debug by listing folder structure
        for directory in [train_label_dir, val_label_dir, train_image_dir, val_image_dir]:
            print(f"\nExamining {directory}:")
            if os.path.exists(directory):
                top_items = os.listdir(directory)[:5]  # Show only first 5 items
                for item in top_items:
                    item_path = os.path.join(directory, item)
                    if os.path.isdir(item_path):
                        print(f"  Folder: {item} - Contains: {os.listdir(item_path)[:5]}...")
                    else:
                        print(f"  File: {item}")
        
        return None, None, None, None, None, None