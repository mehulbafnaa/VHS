"""
Optimized attention mechanisms for dog heart landmark detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

try:
    from flash_attn import flash_attn_qkvpacked_func as flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class RelativePositionEncoding(nn.Module):
    """
    Relative positional encoding for landmark points.
    
    This helps the model understand the circular/sequential arrangement
    of heart landmark points by encoding their relative positions.
    
    Args:
        dim (int): Feature dimension
        num_points (int): Number of landmark points (default: 6)
    """
    def __init__(self, dim, num_points=6):
        super().__init__()
        # Create learnable relative position embeddings
        self.rel_pos_embed = nn.Parameter(torch.zeros(num_points, num_points, dim // 8))
        self.register_buffer('point_indices', torch.arange(num_points))
        nn.init.normal_(self.rel_pos_embed, std=0.02)
    
    def get_rel_pos(self, seq_len):
        """Get relative position encoding."""
        # Calculate relative positions: for each point, what's its position relative to others
        idx = self.point_indices[:seq_len]
        rel_pos_idx = idx.view(-1, 1) - idx.view(1, -1)
        
        # Shift to positive indices
        rel_pos_idx = rel_pos_idx + (seq_len - 1)
        return rel_pos_idx
    
    def forward(self, x):
        """
        Add relative positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, num_points, dim]
            
        Returns:
            x with relative positional encoding
        """
        seq_len = x.size(1)
        rel_pos_idx = self.get_rel_pos(seq_len)
        
        # Get corresponding embeddings
        rel_pos_emb = self.rel_pos_embed[:seq_len, :seq_len]
        
        return x, rel_pos_emb


class EinopsAttention(nn.Module):
    """
    Efficient attention implementation using einops.
    
    Features:
    - Cleaner tensor operations with einops
    - Optional Flash Attention support
    - Support for relative position embeddings
    
    Args:
        dim (int): Feature dimension
        num_heads (int): Number of attention heads
        attn_drop (float): Attention dropout rate
        use_flash_attn (bool): Whether to use Flash Attention when available
    """
    def __init__(self, dim, num_heads=8, attn_drop=0.1, use_flash_attn=HAS_FLASH_ATTN):
        super().__init__()
        assert dim % num_heads == 0, f"Dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn
        
        # Projection layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, q, k, v, rel_pos_emb=None, attn_mask=None):
        """
        Perform attention operation.
        
        Args:
            q: Query tensor [B, Nq, D]
            k: Key tensor [B, Nk, D]
            v: Value tensor [B, Nv, D]
            rel_pos_emb: Optional relative position embeddings
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor after attention
        """
        B, Nq, _ = q.shape
        _, Nk, _ = k.shape
        
        # Project and reshape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        if self.use_flash_attn and Nq == Nk and attn_mask is None and rel_pos_emb is None:
            # Use Flash Attention when available and applicable
            # Flash attention requires qkv packed format
            q = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
            k = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
            v = rearrange(v, 'b n (h d) -> b n h d', h=self.num_heads)
            
            # Pack into [b, seq_len, 3, num_heads, head_dim]
            qkv = torch.stack([q, k, v], dim=2)
            
            # Apply flash attention
            out = flash_attn(qkv)
            
            # Reshape back
            out = rearrange(out, 'b n h d -> b n (h d)')
        else:
            # Standard attention with einops
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads) * self.scale
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
            
            # Compute attention scores
            attn = torch.einsum('bhqd,bhkd->bhqk', q, k)
            
            # Add relative positional embeddings if provided
            if rel_pos_emb is not None:
                # Apply relative position embeddings to attention scores
                # First reshape rel_pos_emb to match attention heads
                rel_pos_emb = repeat(rel_pos_emb, 'q k d -> b h q k d', b=B, h=self.num_heads)
                
                # Compute attention bias from relative positions
                rel_attn = torch.einsum('bhqd,bhqkd->bhqk', q, rel_pos_emb)
                attn = attn + rel_attn
            
            # Apply attention mask if provided
            if attn_mask is not None:
                attn = attn + attn_mask
            
            # Softmax and dropout
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            # Apply attention to values
            out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Final projection
        out = self.out_proj(out)
        return out


class BidirectionalLandmarkAttention(nn.Module):
    """
    Bidirectional attention between landmark points and image features,
    and among landmark points themselves.
    
    This module enables:
    1. Points to attend to image features (visual grounding)
    2. Points to attend to other points (anatomical consistency)
    3. Image features to attend to points (feature enhancement)
    
    Args:
        dim (int): Feature dimension
        num_heads (int): Number of attention heads
        num_points (int): Number of landmark points
        dropout (float): Dropout rate
    """
    def __init__(self, dim, num_heads=8, num_points=6, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        
        # Attention modules
        self.points_to_img = EinopsAttention(dim, num_heads, dropout)
        self.img_to_points = EinopsAttention(dim, num_heads, dropout)
        self.points_to_points = EinopsAttention(dim, num_heads, dropout)
        
        # Relative position encoding for points
        self.rel_pos_encoding = RelativePositionEncoding(dim, num_points)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        
        # FFNs
        self.points_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        self.img_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, points_features, img_features):
        """
        Bidirectional attention between points and image features.
        
        Args:
            points_features: Point query features [B, num_points, D]
            img_features: Image features [B, N, D]
            
        Returns:
            Updated point features and image features
        """
        # Add relative position encoding to point features
        points_features_pos, rel_pos_emb = self.rel_pos_encoding(points_features)
        
        # 1. Points attend to image
        points_tmp = self.points_to_img(
            q=points_features_pos,
            k=img_features,
            v=img_features
        )
        points_features = self.norm1(points_features + points_tmp)
        
        # 2. Points attend to other points (with relative position encoding)
        points_tmp = self.points_to_points(
            q=points_features,
            k=points_features,
            v=points_features,
            rel_pos_emb=rel_pos_emb
        )
        points_features = self.norm2(points_features + points_tmp)
        
        # 3. Image attends to points
        img_tmp = self.img_to_points(
            q=img_features,
            k=points_features,
            v=points_features
        )
        img_features = self.norm3(img_features + img_tmp)
        
        # Apply FFNs
        points_features = self.norm4(points_features + self.points_ffn(points_features))
        
        return points_features, img_features