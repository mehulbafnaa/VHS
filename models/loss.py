"""
Loss functions for dog heart VHS prediction with anatomical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomicalLoss(nn.Module):
    """
    Loss function that incorporates anatomical constraints for dog heart landmarks.
    
    This loss function combines:
    1. Point coordinate prediction loss (SmoothL1Loss)
    2. VHS prediction loss (L1Loss)
    3. Perimeter-to-VHS ratio consistency loss
    4. Shape consistency loss (preserving relative distances between points)
    
    Args:
        points_weight (float): Weight for the points prediction loss
        vhs_weight (float): Weight for the VHS prediction loss
        perimeter_weight (float): Weight for the perimeter-based loss
        shape_weight (float): Weight for the shape consistency loss
        smooth_l1_beta (float): Beta parameter for SmoothL1Loss
    """
    def __init__(self, points_weight=5.0, vhs_weight=1.0, perimeter_weight=0.0, 
                shape_weight=0.0, smooth_l1_beta=0.1):
        super().__init__()
        self.points_weight = points_weight
        self.vhs_weight = vhs_weight
        self.perimeter_weight = perimeter_weight
        self.shape_weight = shape_weight
        self.smooth_l1_beta = smooth_l1_beta
        
        # Loss functions
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=smooth_l1_beta)
        self.l1_loss = nn.L1Loss()
    
    def forward(self, outputs, targets):
        """
        Calculate the loss.
        
        Args:
            outputs (dict): Model outputs
            targets (dict): Target values
            
        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: Combined weighted loss
                - loss_dict: Dictionary with individual loss components
        """
        loss = 0
        losses = {}
        
        # Points prediction loss (smooth L1 for stability)
        if 'points' in outputs and 'points' in targets:
            points_loss = self.smooth_l1_loss(outputs['points'], targets['points'])
            loss += self.points_weight * points_loss
            losses['points_loss'] = points_loss.item()
        
        # VHS prediction loss
        if 'vhs' in outputs and 'vhs' in targets:
            vhs_loss = self.l1_loss(outputs['vhs'], targets['vhs'])
            loss += self.vhs_weight * vhs_loss
            losses['vhs_loss'] = vhs_loss.item()
            
            # Track individual VHS prediction losses
            if 'vhs_direct' in outputs:
                vhs_direct_loss = self.l1_loss(outputs['vhs_direct'], targets['vhs'])
                losses['vhs_direct_loss'] = vhs_direct_loss.item()
                
            if 'vhs_from_points' in outputs:
                vhs_points_loss = self.l1_loss(outputs['vhs_from_points'], targets['vhs'])
                losses['vhs_points_loss'] = vhs_points_loss.item()
        
        # Perimeter ratio consistency loss
        if (self.perimeter_weight > 0 and 
            'perimeter' in outputs and 'perimeter' in targets and 
            'vhs' in outputs and 'vhs' in targets):
            
            # Calculate predicted ratio with epsilon for stability
            pred_ratio = outputs['perimeter'] / (outputs['vhs'].squeeze() + 1e-6)
            
            # Calculate target ratio
            target_ratio = targets['perimeter'] / (targets['vhs'].squeeze() + 1e-6)
            
            # Ratio consistency loss
            ratio_loss = self.l1_loss(pred_ratio, target_ratio)
            loss += self.perimeter_weight * ratio_loss
            losses['ratio_loss'] = ratio_loss.item()
        
        # Shape consistency loss - ensures points form anatomically plausible shape
        if self.shape_weight > 0 and 'points' in outputs and 'points' in targets:
            # Calculate inter-point distances for prediction and target
            pred_points = outputs['points']
            target_points = targets['points']
            batch_size, num_points, _ = pred_points.shape
            
            # Calculate pairwise distances for each batch item
            pred_dists = []
            target_dists = []
            
            for i in range(num_points):
                for j in range(i+1, num_points):
                    # Calculate distances between points i and j
                    pred_dist = torch.sqrt(torch.sum((pred_points[:, i] - pred_points[:, j])**2, dim=1))
                    target_dist = torch.sqrt(torch.sum((target_points[:, i] - target_points[:, j])**2, dim=1))
                    
                    pred_dists.append(pred_dist)
                    target_dists.append(target_dist)
            
            # Stack distances
            pred_dists = torch.stack(pred_dists, dim=1)
            target_dists = torch.stack(target_dists, dim=1)
            
            # Calculate relative distances (normalize by total distance)
            pred_total_dist = torch.sum(pred_dists, dim=1, keepdim=True) + 1e-6
            target_total_dist = torch.sum(target_dists, dim=1, keepdim=True) + 1e-6
            
            pred_rel_dists = pred_dists / pred_total_dist
            target_rel_dists = target_dists / target_total_dist
            
            # Shape loss based on relative distances
            shape_loss = self.l1_loss(pred_rel_dists, target_rel_dists)
            loss += self.shape_weight * shape_loss
            losses['shape_loss'] = shape_loss.item()
        
        losses['total_loss'] = loss.item()
        return loss, losses


class WingLoss(nn.Module):
    """
    Wing loss for landmark detection, as described in:
    "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks"
    
    This loss function is particularly suitable for landmark detection tasks
    as it gives higher gradients to small errors (improving precision)
    while being robust to outliers.
    
    Args:
        omega (float): The point where wing loss transitions from L1 to log loss
        epsilon (float): A small constant to avoid log(0)
    """
    def __init__(self, omega=10.0, epsilon=2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        # Constant for seamless transition between log and L1
        self.C = self.omega - self.omega * torch.log(1 + self.omega / self.epsilon)
    
    def forward(self, pred, target):
        """
        Calculate the wing loss.
        
        Args:
            pred: Predicted coordinates
            target: Target coordinates
            
        Returns:
            Wing loss
        """
        # Calculate absolute difference
        x = torch.abs(pred - target)
        
        # Apply wing loss formula
        loss = torch.where(
            x < self.omega,
            self.omega * torch.log(1 + x / self.epsilon),
            x - self.C
        )
        
        return loss.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for comparing feature representations.
    
    This is a simplified version that can be extended to use
    pre-trained feature extractors for comparing heart structures.
    
    Args:
        feature_extractor: A model that extracts features from images
        weights (list): Weights for different feature levels
    """
    def __init__(self, feature_extractor=None, weights=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.weights = weights if weights is not None else [1.0]
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        Calculate the perceptual loss.
        
        Args:
            pred: Predicted image or features
            target: Target image or features
            
        Returns:
            Perceptual loss
        """
        if self.feature_extractor is None:
            # If no feature extractor is provided, just return L1 loss
            return self.l1_loss(pred, target)
        
        # Extract features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Calculate loss at each feature level
        loss = 0
        for i, (p_feat, t_feat, weight) in enumerate(zip(pred_features, target_features, self.weights)):
            loss += weight * self.l1_loss(p_feat, t_feat)
        
        return loss