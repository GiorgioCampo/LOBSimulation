import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion time step t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformations (scaling and shifting) to the intermediate features.
    """
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.mlp = nn.Linear(cond_dim, input_dim * 2)

    def forward(self, x, cond):
        # cond: (batch, cond_dim)
        # x: (batch, channels, length) for 1D conv or (batch, hidden_dim)
        modulation = self.mlp(cond)
        
        if x.dim() == 3:
            modulation = modulation.unsqueeze(-1)
            
        gamma, beta = modulation.chunk(2, dim=1)
        return gamma * x + beta

class ResidualBlock(nn.Module):
    """
    WaveNet-style residual block with dilated convolutions and gated activations.
    """
    def __init__(self, channels, cond_dim, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=dilation, dilation=dilation)
        self.film = FiLM(channels * 2, cond_dim)
        self.conv_1x1_tanh = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_1x1_sigmoid = nn.Conv1d(channels, channels, kernel_size=1)
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, cond):
        # x: (batch, channels, length)
        # cond: (batch, cond_dim)
        
        # Dilated convolution
        h = self.dilated_conv(x)
        
        # FiLM conditioning
        h = self.film(h, cond)
        
        # Gated activation
        # Split channels for tanh and sigmoid
        h_tanh, h_sigmoid = h.chunk(2, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)
        
        # Final processing
        res = self.res_conv(h)
        skip = self.skip_conv(h)
        
        return (x + res) * math.sqrt(0.5), skip

class DiffVolume(nn.Module):
    """
    DiffVolume: Conditional score-based diffusion model for LOB volume.
    """
    def __init__(self, input_dim, cond_context_dim, hidden_dim=64, n_layers=32, n_heads=4):
        super().__init__()
        self.input_dim = input_dim # K price levels per side * 2 sides
        self.hidden_dim = hidden_dim
        
        # 1. Initial 1x1 Conv + ReLU
        self.init_conv = nn.Conv1d(1, hidden_dim, kernel_size=1)
        
        # 2. Price Level Embeddings (learnable)
        self.price_level_embedding = nn.Embedding(input_dim, hidden_dim)
        
        # 3. Multi-head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        
        # 4. Time Step Embedding (sinusoidal + MLP)
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 5. Conditioning MLP (for t and external context c)
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim + cond_context_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 6. Stack of Residual Blocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, dilation=2**(i % 10))
            for i in range(n_layers)
        ])
        
        # 7. Final Output Sequence
        self.final_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, t, c):
        """
        x: (batch, input_dim) - noised volume snapshot
        t: (batch,) - diffusion step
        c: (batch, cond_context_dim) - external context
        """
        # x is a vector of volume entries across price levels
        # We treat price levels as a sequence dimension for Conv1d/Attention
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        x_in = x.unsqueeze(1)
        
        # Initial projection
        h = F.relu(self.init_conv(x_in)) # (batch, hidden_dim, input_dim)
        
        # Add price level embeddings
        # h: (batch, hidden_dim, input_dim)
        # embedding: (input_dim, hidden_dim) -> (1, input_dim, hidden_dim) -> (batch, input_dim, hidden_dim)
        pos = torch.arange(self.input_dim, device=x.device)
        p_emb = self.price_level_embedding(pos).unsqueeze(0) # (1, input_dim, hidden_dim)
        
        # Align dimensions for addition
        h = h + p_emb.transpose(1, 2)
        
        # Multi-head Self-Attention
        # self_attention wants (batch, seq_len, embed_dim)
        h_attn = h.transpose(1, 2)
        h_attn, _ = self.attention(h_attn, h_attn, h_attn)
        h = h_attn.transpose(1, 2) # (batch, hidden_dim, input_dim)
        
        # Time and context embedding
        t_emb = self.time_embedding(t) # (batch, hidden_dim)
        cond = torch.cat([t_emb, c], dim=1)
        cond = self.cond_mlp(cond) # (batch, hidden_dim)
        
        # Residual blocks
        skip_total = 0
        for layer in self.residual_layers:
            h, skip = layer(h, cond)
            skip_total += skip
        
        # Aggregated skips
        h = skip_total * math.sqrt(1.0 / len(self.residual_layers))
        
        # Final head
        h = F.relu(self.final_conv1(h))
        out = self.final_conv2(h) # (batch, 1, input_dim)
        
        return out.squeeze(1) # (batch, input_dim)
