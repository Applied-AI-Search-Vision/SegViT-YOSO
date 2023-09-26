import torch.nn as nn
import Block

class YOSOHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=12, num_layers=12):
        super(YOSOHead, self).__init__()
        
        # Semantic Branch: A simple MLP head for semantic segmentation
        self.semantic_branch = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Instance Branch: Multiple Transformer Blocks for instance segmentation
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=4.0, 
                qkv_bias=True, 
                qk_scale=None, 
                drop=0.0, 
                attn_drop=0.0, 
                drop_path=0.0
            ) for _ in range(num_layers)
        ])
        
        self.instance_branch = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)  # For each instance class
        )

    def forward(self, x):
        # Semantic Branch
        semantic_output = self.semantic_branch(x)
        
        # Instance Branch
        for block in self.blocks:
            x = block(x)
        instance_output = self.instance_branch(x)
        
        return semantic_output, instance_output
