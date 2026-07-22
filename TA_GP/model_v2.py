"""
TemporalGAT (T-GAT) - Unified architecture for detecting both continuous and escalation attacks.

Key Innovation: TemporalEdgeEncoder replaces DiaglogueEmbeddingProcessModules.
Instead of destroying temporal info via mean/last aggregation, it extracts rich
temporal features through 3 branches:
  1. Temporal Self-Attention: learns which turns matter most
  2. 1D Temporal Convolution: captures local sequential patterns  
  3. Statistical Features: variance, consecutive diffs, trend detection

These branches are fused via a learned gating mechanism that adaptively weights
content (for TA/MA/GP) vs temporal dynamics (for TA_GP escalation).

Drop-in replacement for MyGAT - same input/output interface.
"""

from gat_with_attr_conv import GATwithEdgeConv
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class TemporalEdgeEncoder(nn.Module):
    """Temporal-aware edge feature encoder.
    
    Replaces DiaglogueEmbeddingProcessModules which destroys temporal info.
    Extracts rich temporal features that capture both:
    - CONTENT (what agents say) → important for continuous attacks (TA, MA, GP)
    - DYNAMICS (how agents change) → important for escalation attacks (TA_GP)
    
    Args:
        edge_dim: dimension of per-turn embeddings (384)
        hidden_dim: internal hidden dimension for fusion
        max_turns: maximum number of dialogue turns
        num_heads: number of attention heads for temporal attention
        dropout: dropout rate
    """
    
    def __init__(self, edge_dim, hidden_dim=256, max_turns=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.edge_dim = edge_dim
        self.max_turns = max_turns
        
        # ── Branch 1: Temporal Self-Attention ──
        # Learns which turns are most informative for detection.
        # For escalation: may learn to focus on later (aggressive) turns.
        # For continuous: learns uniform attention (≈ mean).
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, max_turns, edge_dim) * 0.02)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(edge_dim)
        
        # ── Branch 2: 1D Temporal Convolution ──
        # Captures local sequential patterns (e.g., sudden shifts between turns).
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(edge_dim, edge_dim, kernel_size=2, padding=0),
            nn.GELU(),
            nn.Conv1d(edge_dim, edge_dim, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # ── Branch 3: Statistical Features ──
        # Explicit temporal statistics: mean, last, std, max_consecutive_diff.
        # std & max_diff are KEY for escalation detection:
        #   - Benign agents: std ≈ 0 (consistent across turns)
        #   - Escalation attacker: std > 0 (changes across turns)
        self.stats_proj = nn.Sequential(
            nn.Linear(4 * edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # ── Gated Fusion ──
        # Learns to adaptively combine branches.
        # Gate decides how much to rely on temporal dynamics vs content.
        self.gate_net = nn.Sequential(
            nn.Linear(3 * edge_dim, edge_dim),
            nn.Sigmoid()
        )
        self.content_proj = nn.Linear(edge_dim, edge_dim)  # attention branch (content-heavy)
        self.temporal_proj = nn.Linear(2 * edge_dim, edge_dim)  # conv + stats (temporal-heavy)
        
        self.output_norm = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _compute_stats(self, diag_emb):
        """Extract temporal statistical features.
        
        Args:
            diag_emb: (M, T, D)
        Returns:
            stats: (M, 4*D) concatenation of [mean, last, std, max_diff]
        """
        mean_emb = diag_emb.mean(dim=1)        # (M, D)
        last_emb = diag_emb[:, -1, :]           # (M, D)
        
        # Standard deviation across turns - KEY for escalation detection
        if diag_emb.size(1) > 1:
            std_emb = diag_emb.std(dim=1)       # (M, D)
        else:
            std_emb = torch.zeros_like(mean_emb)
        
        # Max consecutive difference - captures largest single-step change
        if diag_emb.size(1) > 1:
            diffs = diag_emb[:, 1:, :] - diag_emb[:, :-1, :]   # (M, T-1, D)
            max_diff = diffs.abs().max(dim=1).values              # (M, D)
        else:
            max_diff = torch.zeros_like(mean_emb)
        
        return torch.cat([mean_emb, last_emb, std_emb, max_diff], dim=-1)  # (M, 4*D)
    
    def forward(self, diag_emb):
        """
        Args:
            diag_emb: (M, T, D) - per-edge temporal embeddings
        Returns:
            edge_feat: (M, D) - rich temporal edge features
        """
        M, T, D = diag_emb.shape
        
        # Add positional encoding for temporal ordering
        pos_emb = self.temporal_pos_emb[:, :T, :]  # handle variable turns
        diag_emb_pos = diag_emb + pos_emb
        
        # Branch 1: Temporal Self-Attention
        attn_out, _ = self.temporal_attn(diag_emb_pos, diag_emb_pos, diag_emb_pos)
        attn_out = self.attn_norm(attn_out + diag_emb_pos)  # residual connection
        attn_feat = attn_out.mean(dim=1)  # (M, D)
        
        # Branch 2: Temporal Convolution
        conv_in = diag_emb.transpose(1, 2)  # (M, D, T)
        if T < 2:
            # Pad if only 1 turn (conv needs at least kernel_size=2)
            conv_in = F.pad(conv_in, (0, 1))
        conv_feat = self.temporal_conv(conv_in).squeeze(-1)  # (M, D)
        
        # Branch 3: Statistical Features
        stats = self._compute_stats(diag_emb)  # (M, 4*D)
        stats_feat = self.stats_proj(stats)     # (M, D)
        
        # Gated Fusion
        all_feats = torch.cat([attn_feat, conv_feat, stats_feat], dim=-1)  # (M, 3*D)
        gate = self.gate_net(all_feats)  # (M, D) ∈ [0, 1]
        
        content = self.content_proj(attn_feat)                                     # (M, D)
        temporal = self.temporal_proj(torch.cat([conv_feat, stats_feat], dim=-1))   # (M, D)
        
        # gate=1 → rely on content; gate=0 → rely on temporal dynamics
        edge_feat = gate * content + (1 - gate) * temporal
        edge_feat = self.output_norm(self.dropout(edge_feat))
        
        return edge_feat


class TemporalGAT(nn.Module):
    """Temporal-aware GAT for multi-agent attacker detection.
    
    Drop-in replacement for MyGAT. Same __init__ signature (with extra args)
    and same forward(x, edge_index, edge_attr) interface.
    
    Key difference: uses TemporalEdgeEncoder instead of mean/last aggregation,
    preserving temporal dynamics that are critical for escalation attack detection.
    
    Args:
        in_channels: node feature dimension (384)
        hidden_channels: GNN hidden dimension (1024)
        out_channels: output dimension (1 for binary classification)
        heads: number of GAT attention heads
        concat: whether to concatenate multi-head outputs
        edge_dim: tuple of (max_turns, embedding_dim), e.g. (3, 384)
        num_layers: number of GNN layers
        dropout: dropout rate
        residual: whether to use residual connections in GATConv
        temporal_heads: number of heads for temporal self-attention
        temporal_hidden: hidden dim for temporal encoder fusion
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 heads=1, concat=True, edge_dim=None, num_layers=2, 
                 dropout=0.2, residual=False,
                 temporal_heads=4, temporal_hidden=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.heads = heads
        self.head_channels = hidden_channels // heads
        self.hidden_channels = self.head_channels * heads
        
        max_turns, emb_dim = self.edge_dim
        
        # ── Temporal Edge Encoder (replaces DiaglogueEmbeddingProcessModules) ──
        self.temporal_encoder = TemporalEdgeEncoder(
            edge_dim=emb_dim,
            hidden_dim=temporal_hidden,
            max_turns=max_turns,
            num_heads=temporal_heads,
            dropout=dropout
        )
        
        # ── GNN Layers (same as original MyGAT) ──
        self.convs = nn.ModuleList()
        conv1 = GATwithEdgeConv(
            in_channels, self.head_channels,
            heads=heads, concat=concat,
            edge_dim=emb_dim, residual=residual
        )
        self.convs.append(conv1)
        for i in range(num_layers - 1):
            conv_i = GATwithEdgeConv(
                self.hidden_channels, self.head_channels,
                heads=heads, concat=concat,
                edge_dim=hidden_channels, residual=residual
            )
            self.convs.append(conv_i)
        
        # ── Output Head ──
        self.out = nn.Linear(self.hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (N, in_channels) - node features
            edge_index: (2, M) - edge connectivity
            edge_attr: (M, T, D) - temporal edge attributes
        Returns:
            out: (N, out_channels) - per-node predictions
        """
        # Temporal encoding: (M, T, D) → (M, D) with preserved temporal info
        edge_attr = self.temporal_encoder(edge_attr)
        
        # Standard GNN message passing
        for i in range(self.num_layers):
            x, edge_attr = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.out(x)
        return x


# ── Backward-compatible alias ──
# Can be used as drop-in replacement: from model_v2 import MyGAT
MyGAT = TemporalGAT


if __name__ == "__main__":
    # Quick sanity check
    batch_edges = 10
    num_nodes = 4
    num_turns = 3
    emb_dim = 384
    
    x = torch.randn(num_nodes, emb_dim)
    edge_index = torch.randint(0, num_nodes, (2, batch_edges))
    edge_attr = torch.randn(batch_edges, num_turns, emb_dim)
    
    model = TemporalGAT(
        in_channels=emb_dim,
        hidden_channels=1024,
        out_channels=1,
        heads=8,
        edge_dim=(num_turns, emb_dim),
        temporal_heads=4,
        temporal_hidden=256
    )
    
    out = model(x, edge_index, edge_attr)
    print(f"Input:  x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}")
    print(f"Output: {out.shape}")
    
    # Parameter count comparison
    from model import MyGAT as OriginalGAT
    original = OriginalGAT(emb_dim, 1024, 1, heads=8, edge_dim=(num_turns, emb_dim))
    
    orig_params = sum(p.numel() for p in original.parameters())
    new_params = sum(p.numel() for p in model.parameters())
    print(f"\nOriginal MyGAT params: {orig_params:,}")
    print(f"TemporalGAT params:   {new_params:,}")
    print(f"Overhead:             {new_params - orig_params:,} ({(new_params/orig_params - 1)*100:.1f}%)")
