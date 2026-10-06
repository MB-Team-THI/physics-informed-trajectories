import math
import torch
import torch.nn as nn
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xformers.ops import memory_efficient_attention
import xformers.ops as xops


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, pos: Optional[int] = None) -> torch.Tensor:
        """
        x: [T, B, D] – token embeddings
        pos: int or None – specific time step index (e.g., for stepwise decoding)
        """
        if pos is not None:
            pe_slice = self.pe[pos:pos+1]  # [1, D]
            x = x + pe_slice.unsqueeze(1)  # [1, B, D]
        else:
            T = x.size(0)
            x = x + self.pe[:T].unsqueeze(1)
        return x



import torch
import torch.nn as nn
from typing import Optional

class TrajectoryTransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 5,      # X, Y, T, ψ, v
        model_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
        out_dim: int = 64,       # yaw_rate & a_x
        max_len: int = 256,
        enc_dim: int = 256 + 2,
        cross_attention: bool = True,
        sampling_mode: str = 'continuous'
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.cond_proj = nn.Linear(2, model_dim)
        self.pos_enc = SinusoidalPosEnc(model_dim, max_len)
        self.sampling_mode = sampling_mode
        self.use_cross_attn = cross_attention

        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=True
        )

        self.out_proj = nn.Linear(model_dim, 2 * out_dim)

    def forward(
        self,
        inputs_embeds: torch.Tensor,         # tgt [T, B, input_dim]
        position_ids: torch.LongTensor = None,
        condition: Optional[torch.Tensor] = None  # src [S, B, enc_dim]
    ) -> torch.Tensor:

        T, B, _ = inputs_embeds.shape

        tgt = self.input_proj(inputs_embeds)               # [T, B, D]
        tgt = self.pos_enc(tgt)                            # [T, B, D]

        if self.use_cross_attn and condition is not None:
            src = self.cond_proj(condition)                # [S, B, D]
            src = self.pos_enc(src)
        else:
            src = torch.zeros(1, B, tgt.size(-1), device=tgt.device)  # dummy src

        # (optional) tgt_mask for autoregressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        out = self.transformer(
            src=src, tgt=tgt,
            tgt_mask=tgt_mask
        )  # [T, B, D]

        return self.out_proj(out).transpose(1,0)  # [T, B, out_dim*2]


class DecoderBlock(nn.Module):
    """
    A single Pre-LN decoder block with xformers memory-efficient causal attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal_mask = xops.LowerTriangularMask()

        # LayerNorms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # qkv and output projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


        # simple 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        B, S, E = x.shape

        # --- self-attention ---
        residual = x
        x_norm = self.ln1(x)
        qkv = self.qkv_proj(x_norm)       # (B, S, 3E)
        q, k, v = qkv.chunk(3, dim=-1)    # each (B, S, E)

        # reshape for multi-head
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, S, d)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = multihead_attention(q,k,v,causal=True, dropout_p=self.dropout)
        # WARNING maybe still broken!!!!!
        # attn_out = memory_efficient_attention(q, k, v,attn_bias=self.causal_mask, p=self.dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, E)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out

        # --- MLP ---
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x








class DecoderOnlyContinuousModel(nn.Module):
    """
    Decoder-only Transformer that expects continuous embeddings as input.
    """
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        sampling_mode="gumbel"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.sampling_mode = sampling_mode
        self.input_projector = nn.Linear(5, embed_dim)

        # learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.cond_emb = nn.Linear(2, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # stack of decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # final norm + LM head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size * 2, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        inputs_embeds: (B, S, E)
        position_ids:   (B, S), optional
        returns logits (B, S, V)
        """
        inputs_embeds = rearrange(inputs_embeds, "seq B dim -> B seq dim")
        B, S, E = inputs_embeds.size()
        device = inputs_embeds.device
        inputs_embeds = self.input_projector(inputs_embeds)

        if condition is not None:
            cond_emb = self.cond_emb(condition)
            cond_emb = rearrange(cond_emb, "1 seq dim  -> seq 1 dim")
            # input_embdes 87,20,128
            inputs_embeds = torch.cat([cond_emb, inputs_embeds], dim =1)
            S+=1

        # build position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(position_ids)  # (B, S, E)
        x = self.dropout(inputs_embeds + pos)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, S, V)
        return logits

class CrossDecoderBlock(nn.Module):
    """
    Pre-LN decoder block with (1) causal self-attention, (2) cross-attention to an encoder sequence, (3) MLP.
    Each block has its own cross-attn projections (q/k/v), so it can reshape the encoder context per layer.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Norms
        self.ln_self = nn.LayerNorm(embed_dim)
        self.ln_cross = nn.LayerNorm(embed_dim)
        self.ln_mlp = nn.LayerNorm(embed_dim)

        # Self-attention projections
        self.self_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.self_out = nn.Linear(embed_dim, embed_dim)

        # Cross-attention projections (query from decoder, key/value from encoder)
        self.cross_q = nn.Linear(embed_dim, embed_dim)
        self.cross_k = nn.Linear(embed_dim, embed_dim)
        self.cross_v = nn.Linear(embed_dim, embed_dim)
        self.cross_out = nn.Linear(embed_dim, embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E) -> (B, h, S, d)
        B, S, E = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, S_dec, E)   - decoder hidden states
        enc: (B, S_enc, E)   - encoder hidden states (already projected to E if needed)
        """
        B, S_dec, E = x.shape
        enc, enc_mask = enc
        # ---- Self-attention (causal) ----
        residual = x
        xs = self.ln_self(x)
        qkv = self.self_qkv(xs)                                  # (B, S_dec, 3E)
        q, k, v = qkv.chunk(3, dim=-1)                           # each (B, S_dec, E)
        q = self._reshape_heads(q); k = self._reshape_heads(k); v = self._reshape_heads(v)
        self_out = multihead_attention(q, k, v, causal=True, dropout_p=self.dropout)
        self_out = self_out.transpose(1, 2).contiguous().view(B, S_dec, E)
        x = residual + self.self_out(self_out)

        # ---- Cross-attention (non-causal) ----
        residual = x
        xc = self.ln_cross(x)
        # queries from decoder, keys/values from encoder
        q = self._reshape_heads(self.cross_q(xc))                # (B, h, S_dec, d)
        k = self._reshape_heads(self.cross_k(enc))               # (B, h, S_enc, d)
        v = self._reshape_heads(self.cross_v(enc))               # (B, h, S_enc, d)

        cross_out, cross_attn_weights = multihead_attention(
            q, k, v, causal=False, dropout_p=self.dropout,
            key_padding_mask=enc_mask, return_weights=True
        )
        cross_out = cross_out.transpose(1, 2).contiguous().view(B, S_dec, E)
        x = residual + self.cross_out(cross_out)

        # ---- MLP ----
        residual = x
        xm = self.ln_mlp(x)
        x = residual + self.mlp(xm)
        return x, cross_attn_weights
def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    dropout_p: float = 0.0,
    key_padding_mask: Optional[torch.Tensor] = None,  # (B, S_k) True => ignore
    use_sdpa: bool = False,
    return_weights = False
) -> torch.Tensor:
    """
    Multi-head attention with optional PyTorch SDPA backend.

    Args:
        q, k, v: (B, h, S, d)
        causal: apply causal mask
        dropout_p: dropout on attention weights (training only)
        key_padding_mask: (B, S_k) where True marks positions to ignore
        use_sdpa: if True, use torch.nn.functional.scaled_dot_product_attention

    Returns:
        attn_output: (B, h, S, d)
    """
    B, h, S_q, d = q.shape
    S_k = k.shape[-2]
    training = q.requires_grad  # simple proxy; pass your own flag if needed

    if use_sdpa:
        # SDPA expects attn_mask broadcastable to (B, h, S_q, S_k).
        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, S_k) -> broadcast over heads and query length
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)

        # Pass is_causal to get an efficient triangular mask; attn_mask (if provided)
        # is combined with it internally. Dropout only when "training".
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,              # boolean mask: True => masked
            dropout_p=dropout_p if training else 0.0,
            is_causal=causal
        )

    # -------- Manual path (reference / fallback) --------
    scale = d ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, h, S_q, S_k]

    if key_padding_mask is not None:
        mask = key_padding_mask[:, None, None, :].to(torch.bool)  # [B,1,1,S_k]
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    if causal:
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=q.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_scores, dim=-1)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    attn_output = torch.matmul(attn_weights, v)  # [B, h, S_q, d]
    if return_weights:
        return attn_output, attn_weights
    return attn_output

class DecoderWithEncoderCrossAttention(nn.Module):
    """
    Transformer decoder that cross-attends to an external encoder sequence.
    - Decoder inputs are continuous features (e.g., your 5-dim token), projected to embed_dim.
    - Encoder sequence is provided as a tensor and (optionally) projected to embed_dim.

    Args:
        embed_dim: model width
        max_seq_len: for learned positional embeddings on decoder side
        vocab_size: output head size factor (kept consistent with your existing API: *2)
        num_layers, num_heads, dropout: standard Transformer knobs
        enc_dim: dimension of the provided encoder sequence
        project_encoder: if True and enc_dim != embed_dim, adds a linear to map encoder to embed_dim
    """
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        enc_dim: int = 256,
        project_encoder: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Decoder input projection (your 5-dim continuous token -> E)
        self.input_projector = nn.Linear(5, embed_dim)

        # Positional embeddings for decoder tokens
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional encoder projector to match dimensions
        self.enc_proj = None
        if project_encoder or enc_dim != embed_dim:
            self.enc_proj = nn.Linear(enc_dim, embed_dim)

        # Stack of cross-attending decoder blocks
        self.blocks = nn.ModuleList([
            CrossDecoderBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size * 2, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,          # (seq_dec, B, feat=5) or (B, S_dec, 5) – we accept your earlier shape
        encoder_seq: torch.Tensor,            # (B, S_enc, enc_dim)
        position_ids: Optional[torch.LongTensor] = None,
        visualize_attention: bool = False
    ) -> torch.Tensor:
        # Normalize decoder input shape to (B, S_dec, 5)
        if inputs_embeds.dim() == 3 and inputs_embeds.shape[0] != inputs_embeds.shape[1]:
            # Likely (seq, B, dim) as you used before -> make (B, seq, dim)
            x = rearrange(inputs_embeds, "seq B dim -> B seq dim")
        else:
            x = inputs_embeds  # assume (B, S, dim)
        B, S_dec, _ = x.shape
        device = x.device

        # Project decoder inputs and add positions
        x = self.input_projector(x)  # (B, S_dec, E)
        if position_ids is None:
            position_ids = torch.arange(S_dec, device=device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(position_ids)      # (B, S_dec, E)
        x = self.dropout(x + pos)

        # Prepare encoder sequence to model width
        enc = encoder_seq

        for i, block in enumerate(self.blocks):
            # MODIFICATION 1: Unpack the returned attention weights
            x, attn_weights = block(x, enc)

            # MODIFICATION 2: Add the plotting logic
            if visualize_attention:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Process weights for plotting: use first batch item, average heads
                attn_map = attn_weights[0].mean(dim=0).detach().cpu().numpy()

                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(attn_map, cmap='cool')
                ax.set_title(f"Cross-Attention Heatmap (Layer {i + 1})")
                ax.set_xlabel("Encoder Sequence")
                ax.set_ylabel("Decoder Sequence")
                plt.show()

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, S_dec, vocab_size*2)
        return logits