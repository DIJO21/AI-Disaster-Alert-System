import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class GatedLinearUnit(nn.Module):
    """GLU component for selective information routing."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        val, gate = torch.chunk(x, 2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """GRN component for adaptive non-linear mapping and skipping."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        
        # Projection for skip connection if input and output dimensions mismatch
        self.skip_project = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_project(x)
        h = self.elu(self.linear_1(x))
        h = self.linear_2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.layernorm(h + residual)

class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Attention layer to prevent leakage of future info."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Create upper triangular causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        
        attn_out, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        return self.norm(x + self.dropout(attn_out))

class TemporalForecaster(nn.Module):
    """
    NASA-grade Temporal Fusion Transformer-inspired forecaster.
    Supports causal self-attention, probabilistic quantile forecasts, and uncertainty estimation.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        
        # Project inputs to hidden state representation
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Variable Selection / Feature gating representation
        self.grn_input = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        # Causal Attention layer
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        
        # Output gating and residual addition
        self.grn_output = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.output_glu = GatedLinearUnit(hidden_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Multiple quantile output projection heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in quantiles
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (Batch_Size, Seq_Len, Input_Size)
        Output shape: (Batch_Size, Seq_Len, Num_Quantiles)
        """
        # Step 1: Input projection
        h = self.input_projection(x)
        
        # Step 2: Feature gating
        h = self.grn_input(h)
        
        # Step 3: Causal multi-head temporal self-attention
        attn_out = self.attn(h)
        
        # Step 4: Output representation
        out_features = self.grn_output(attn_out)
        gated_out = self.output_glu(out_features)
        rep = self.output_norm(gated_out + h)
        
        # Step 5: Quantile predictions projection
        outputs = [head(rep) for head in self.quantile_heads]
        return torch.cat(outputs, dim=-1) # (Batch, Seq, Num_Quantiles)

    def forecast_with_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_runs: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Performs Monte Carlo Dropout forecasting at inference time.
        Enables Dropout layers manually to estimate prediction confidence intervals.
        """
        # Enable dropout layers specifically
        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
                
        self.eval()
        self.apply(enable_dropout)
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_mc_runs):
                pred = self.forward(x) # (B, S, Q)
                predictions.append(pred.unsqueeze(0))
                
        # Stack predictions: (MC_runs, B, S, Q)
        preds_stacked = torch.cat(predictions, dim=0)
        
        # Extract mean, standard deviation across MC runs
        mean_forecast = preds_stacked.mean(dim=0)
        std_forecast = preds_stacked.std(dim=0)
        
        # Map back to standard eval mode
        self.eval()
        
        return {
            "forecast": mean_forecast,
            "uncertainty": std_forecast,
            "mc_samples": preds_stacked
        }

class QuantileLoss(nn.Module):
    """Pinball / Quantile Loss function for probabilistic forecasts."""
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds shape: (B, S, Q)
        targets shape: (B, S, 1) or (B, S) - needs to match spatial size of preds
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
            
        losses = []
        for i, q in enumerate(self.quantiles):
            error = targets - preds[..., i:i+1]
            # Pinball loss formula: max(q * error, (q - 1) * error)
            q_loss = torch.max(q * error, (q - 1) * error)
            losses.append(q_loss.mean())
            
        return torch.stack(losses).sum()
