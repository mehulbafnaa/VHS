"""
Vision Transformer with optimized attention for dog heart VHS prediction.
"""

import torch
import torch.nn as nn
import timm

from .landmark_attention import BidirectionalLandmarkAttention


class DogHeartViTWithAttention(nn.Module):
    """
    Vision Transformer with optimized bidirectional attention for landmark points.
    
    This model uses a pre-trained ViT backbone to extract features from dog heart X-rays,
    then applies a specialized attention mechanism to predict anatomical landmarks
    and Vertebral Heart Score (VHS).
    
    Features:
    - ViT backbone for feature extraction
    - Bidirectional attention for landmark point prediction
    - Dual path VHS prediction (direct and point-based)
    
    Args:
        pretrained (bool): Whether to use pre-trained weights for backbone
        freeze_backbone (bool): Whether to freeze backbone weights
        backbone (str): Which ViT model to use
        num_points (int): Number of landmark points to predict
        img_size (int): Input image size
    """
    def __init__(self, pretrained=True, freeze_backbone=True, backbone='vit_base_patch16_224',
                num_points=6, img_size=224):
        super().__init__()
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        if 'base' in backbone:
            feature_dim = 768
        elif 'small' in backbone:
            feature_dim = 384
        elif 'large' in backbone:
            feature_dim = 1024
        else:
            feature_dim = 768  # Default
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature projection for image features
        self.img_projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Initial point query embeddings (learnable)
        self.point_queries = nn.Parameter(torch.randn(1, num_points, 256))
        nn.init.normal_(self.point_queries, std=0.02)
        
        # Bidirectional attention module
        self.attn_module = BidirectionalLandmarkAttention(
            dim=256,
            num_heads=8,
            num_points=num_points
        )
        
        # Output heads for points
        self.points_norm = nn.LayerNorm(256)
        self.points_x_head = nn.Linear(256, 1)  # x-coordinate
        self.points_y_head = nn.Linear(256, 1)  # y-coordinate
        
        # VHS prediction from image features
        self.vhs_direct = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # VHS prediction from predicted points
        self.vhs_from_points = nn.Sequential(
            nn.Linear(num_points * 2, 64),  # num_points * 2 coordinates
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Store image size info
        self.img_size = img_size
        self.num_points = num_points
        
        # Initialize weights of new layers
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Dictionary with predictions:
            - points: Predicted landmark points [batch_size, num_points, 2]
            - perimeter: Perimeter of the predicted points [batch_size]
            - vhs_direct: VHS prediction from image features [batch_size, 1]
            - vhs_from_points: VHS prediction from points [batch_size, 1]
            - vhs: Final VHS prediction [batch_size, 1]
        """
        # Get image features from ViT backbone
        batch_size = x.shape[0]
        features = self.backbone(x)
        
        # Project image features
        img_features = self.img_projection(features).unsqueeze(1)  # [B, 1, 256]
        
        # Initialize point queries with learned embeddings
        point_queries = self.point_queries.expand(batch_size, -1, -1)  # [B, num_points, 256]
        
        # Apply bidirectional attention
        points_features, _ = self.attn_module(point_queries, img_features)
        
        # Apply final normalization
        points_features = self.points_norm(points_features)
        
        # Predict x,y coordinates for each point
        # Outputs normalized coordinates [0,1] and scale to image dimensions
        # x_coords = self.points_x_head(points_features).squeeze(-1) * self.img_size  # [B, num_points]
        # y_coords = self.points_y_head(points_features).squeeze(-1) * self.img_size  # [B, num_points]
        
        x_coords = torch.sigmoid(self.points_x_head(points_features).squeeze(-1)) * self.img_size  # [B, num_points]
        y_coords = torch.sigmoid(self.points_y_head(points_features).squeeze(-1)) * self.img_size  # [B, num_points]

        x_coords = self.img_size - x_coords
        y_coords = self.img_size - y_coords

        # Stack x,y coordinates
        points_pred = torch.stack([x_coords, y_coords], dim=2)  # [B, num_points, 2]
        
        # Predict VHS directly from image features
        vhs_direct_pred = self.vhs_direct(features)  # [B, 1]
        
        # Predict VHS from predicted points
        points_flat = points_pred.reshape(batch_size, -1)  # [B, num_points*2]
        vhs_from_points_pred = self.vhs_from_points(points_flat)  # [B, 1]
        
        # Combine VHS predictions (weighted average)
        alpha = 0.7  # Weight for direct prediction
        vhs_pred = alpha * vhs_direct_pred + (1 - alpha) * vhs_from_points_pred
        
        # Calculate perimeter for loss function
        perimeter = torch.zeros(batch_size, device=x.device)
        for i in range(self.num_points):
            next_i = (i + 1) % self.num_points
            p1 = points_pred[:, i, :]
            p2 = points_pred[:, next_i, :]
            dist = torch.sqrt(torch.sum((p2 - p1) ** 2, dim=1))
            perimeter += dist
        
        # Return predictions
        return {
            'points': points_pred,  # [B, num_points, 2]
            'perimeter': perimeter,  # [B]
            'vhs_direct': vhs_direct_pred,  # [B, 1]
            'vhs_from_points': vhs_from_points_pred,  # [B, 1]
            'vhs': vhs_pred  # [B, 1]
        }