
import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder
import random
from einops import rearrange, reduce
class lstm_decoder(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, num_layers=1, use_condition=False):
        super(lstm_decoder, self).__init__()
        raise NotImplementedError("EMbedding dim not defined")
        self.hidden_cond_size = 8
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vor_linear = nn.Linear(hidden_size, hidden_size)
        self.use_condition = use_condition
        new_input_size = input_size + self.hidden_cond_size if use_condition else input_size

        self.lstm = nn.LSTM(input_size=new_input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.nach_linear = nn.Linear(hidden_size, output_size)
        self.condition_encoder = nn.Linear(0, self.hidden_cond_size)

    def forward(self, x_input, encoder_hidden_states, t_steps, conditions):
        if not self.use_condition:
            conditions = None

        if self.use_condition:
            enc_conditions = self.condition_encoder(conditions)
            enc_conditions = rearrange(enc_conditions, '1 b dim -> b 1 dim')
            x_input = torch.cat((x_input, enc_conditions), dim=2)

        if t_steps == 0:
            if not self.use_condition:
                # Ensure correct hidden state structure even when conditions are not used
                encoder_hidden_states = (encoder_hidden_states[0], encoder_hidden_states[0])
            output, self.hidden = self.lstm(x_input, encoder_hidden_states)
        else:
            output, self.hidden = self.lstm(x_input, encoder_hidden_states)

        output = self.nach_linear(output)
        return output, self.hidden

class SimpleImageEncoder(nn.Module):
    def __init__(self, channels, z_dim_i):
        super(SimpleImageEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Convolutional Block 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Convolutional Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Convolutional Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Adaptive pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layer
        self.fc = nn.Linear(256, z_dim_i)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class merger_encoder(nn.Module):
    def __init__(self, *, dim_in=64, dim_trans=128, depth=6, heads=8, condition):
        super().__init__()
        self.dim = dim_trans
        self.model = ContinuousTransformerWrapper(
            dim_in=dim_in,
            max_seq_len=150,
            use_abs_pos_emb=False,
            attn_layers=Encoder(
                dim=dim_trans,
                depth=depth,
                heads=heads,
                cross_attend=condition
            )
        )

    def forward(self, x_infra, x_obj, obj_length_sum, obj_length, dim_i=64, conditions=None):

        scenario_wise_merge = []
        # unpack objects and create scenario wise merge
        for i in range(len(obj_length)):
            scenario_wise_merge_local = []
            if x_infra is not None:
                scenario_wise_merge_local.append(x_infra[i].unsqueeze(0))
            scenario_wise_merge_local.append(x_obj[obj_length_sum[i]:obj_length_sum[i + 1], :])
            scenario_wise_merge_local = [j for i in scenario_wise_merge_local for j in i]
            scenario_wise_merge_local = torch.stack(scenario_wise_merge_local, dim=0)  # (infra + obj) x emb_dim
            scenario_wise_merge.append(scenario_wise_merge_local)
        scenario_wise_merge = torch.nn.utils.rnn.pad_sequence(scenario_wise_merge,
                                                              batch_first=True)  # batch_size x max(infra + obj) x emb_dim
        # scenario_wise_merge = scenario_wise_merge.to(x_infra.device) # write comment here why the hell I did
        if x_infra is not None:
            obj_length = torch.Tensor(obj_length) + 1  # added infra length
        else:
            obj_length = torch.Tensor(obj_length)
        mask = (torch.arange(scenario_wise_merge.shape[1])[None, :] < obj_length[:, None]).to(
            x_obj.device)  # ,device = scenario_wise_merge.device # mask out irrelevant objects

        x = self.model(scenario_wise_merge, mask=mask)  # batch_size x max(infra + obj) x emb_dim
        return x, obj_length

class traj_encoder(nn.Module):
    def __init__(self, *, dim_in, dim_out, depth, heads, dim_trans, dim_mlp):
        super().__init__()
        self.emb_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.model = ContinuousTransformerWrapper(
            dim_in=dim_in,
            max_seq_len=51,
            use_abs_pos_emb=True,
            attn_layers=Encoder(
                dim=dim_trans,
                depth=depth,
                heads=heads
            )
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(dim_trans, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, dim_out)
        )

    def forward(self, x):
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        emb_tokens = self.emb_token.expand(seq_unpacked.shape[0], -1, -1)
        seq_unpacked = torch.cat((emb_tokens, seq_unpacked), dim=1)
        mask = (torch.arange(seq_unpacked.shape[1])[None, :] < lens_unpacked[:, None] + 1).to(seq_unpacked.device)
        x = self.model(seq_unpacked, mask=mask)
        return self.mlp_head(x[:, 0, :])

def activate_soft_clip(x, lower_bound, upper_bound):
    """
    Softly clip x to [lower_bound, upper_bound] via a sigmoid.
    """
    sigmoid = torch.sigmoid(x)
    return lower_bound + (upper_bound - lower_bound) * sigmoid


def activate_psi_dot(raw_yaw_rate, velocity):
    """
    Convert raw_yaw_rate into a physically valid yaw_rate by capping lateral acceleration.
    lateral_acc = raw_yaw_rate * velocity, clipped to [-8, 8].
    Then yaw_rate = clipped_lateral_acc / (velocity + epsilon).
    """
    lateral_acc = raw_yaw_rate * velocity
    lateral_acc_clipped = activate_soft_clip(lateral_acc, -1.0, 1.0)

    # Small epsilon to avoid division by zero:
    epsilon = 1e-3
    v_safe = velocity + epsilon

    return lateral_acc_clipped / v_safe

def top_k_sampling(probs, bins, k):
    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # renormalize
    sampled_idx = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
    return bins[topk_indices[torch.arange(probs.size(0)), sampled_idx]]


def top_p_sampling(probs, bins, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask where cumulative prob exceeds p
    mask = cum_probs > p
    # Keep at least one element
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    # Masked probs and renormalization
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    final_indices = sorted_indices[torch.arange(probs.size(0)), sampled_idx]
    return bins[final_indices]


def normalize_except_zero(t: torch.Tensor, val_max=211.625, val_min=-137.875):
    # Erstelle eine Maske für alle Werte, die nicht 0 sind
    mask = t != 0

    # Normalisiere nur die Werte, die nicht 0 sind
    normalized_t = torch.where(mask, (t - val_min) / (val_max - val_min), t)

    return normalized_t

def debug_tensor(t: torch.Tensor, name="tensor", step=None, save_hist=False):
    print("------ Debug tensor stats ------")
    print(f"[{name}] device={t.device}, dtype={t.dtype}")
    with torch.no_grad():
        t_det = t.detach()
        shape = tuple(t_det.shape)
        numel = t_det.numel()

        is_nan = torch.isnan(t_det)
        is_pinf = torch.isposinf(t_det)
        is_ninf = torch.isneginf(t_det)
        is_finite = torch.isfinite(t_det)

        n_nan = int(is_nan.sum().item())
        n_pinf = int(is_pinf.sum().item())
        n_ninf = int(is_ninf.sum().item())
        n_finite = int(is_finite.sum().item())

        msg = (f"[{name}] step={step} shape={shape} numel={numel} | "
               f"finite={n_finite} ({n_finite / numel:.2%}), "
               f"nan={n_nan}, +inf={n_pinf}, -inf={n_ninf}")
        print(msg)

        fin = t_det[is_finite]
        if fin.numel() > 0:
            mn = fin.min().item()
            med = fin.median().item()
            mean = fin.mean().item()
            std = fin.std(unbiased=False).item()
            mx = fin.max().item()
            print(
                f"[{name}] min/median/mean/std/max = {mn:.6g} / {med:.6g} / {mean:.6g} / {std:.6g} / {mx:.6g}")

            # Percentiles (nice to catch heavy tails)
            qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0], device=fin.device)
            fin = fin.to(torch.float32)  # quantile() doesn't support float16
            pct = torch.quantile(fin.flatten(), qs)
            print(f"[{name}] percentiles {qs.tolist()}: {pct.detach().cpu().numpy()}")

            # Histogram (clipped to avoid plot domination by outliers)
            if save_hist:
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    fin_np = fin.flatten().cpu().numpy()
                    lo = np.percentile(fin_np, 0.5)
                    hi = np.percentile(fin_np, 99.5)
                    clipped = np.clip(fin_np, lo, hi)

                    fig = plt.figure()
                    plt.hist(clipped, bins=100)
                    plt.title(f"{name} histogram (clipped 0.5- 99.5 pct)")
                    plt.xlabel(name)
                    plt.ylabel("count")
                    plt.tight_layout()
                    fname = f"{name}_hist_step{step if step is not None else 'x'}.png"
                    plt.show()
                    print(f"[{name}] histogram saved to {fname}")
                except Exception as e:
                    print(f"[{name}] plotting failed: {e}")
        else:
            print(f"[{name}] no finite values; skip stats/hist.")

        # Show a few bad indices to chase upstream
        bad = ~is_finite
        if bad.any():
            idx = torch.nonzero(bad, as_tuple=False)
            k = min(8, idx.size(0))
            sl = idx[:k]
            print(f"[{name}] example non-finite indices (up to {k}): {sl.tolist()}")

def trace_autograd_graph(tensor, max_depth=500):
    """
    Recursively print the autograd graph of a tensor.
    Useful to check whether the computation graph is still attached
    or if the tensor was detached somewhere.

    Args:
        tensor: torch.Tensor
        max_depth: how deep to traverse (to avoid infinite recursion)
    """
    def _recurse(fn, depth, visited):
        indent = "  " * depth
        if fn in visited or fn is None:
            return
        visited.add(fn)
        print(f"{indent}{type(fn).__name__}")
        if depth >= max_depth:
            return
        for next_fn, _ in getattr(fn, 'next_functions', []):
            _recurse(next_fn, depth + 1, visited)

    if not isinstance(tensor, torch.Tensor):
        print("Input is not a tensor.")
        return
    if tensor.grad_fn is None:
        print("⚠️  Tensor has no grad_fn — it's likely detached or a leaf.")
        return
    print(f"Autograd graph for tensor {tuple(tensor.shape)}:")
    visited = set()
    _recurse(tensor.grad_fn, depth=0, visited=visited)


import torch

def compute_uncertainty_metrics(
    logits_acc: torch.Tensor,
    NUM_BINS_YAW: int,
    NUM_BINS_AX: int,
    bin_centers_yaw: torch.Tensor | None = None,  # [K_yaw], optional
    bin_centers_ax:  torch.Tensor | None = None,  # [K_ax],  optional
):
    """
    logits_acc: [T, N, K_yaw+K_ax] raw logits (no softmax)
    Returns dict with per-(t,n) metrics and convenient aggregations.
    """
    T, N, Ktot = logits_acc.shape
    assert Ktot == NUM_BINS_YAW + NUM_BINS_AX

    # Split
    logits_yaw = logits_acc[..., :NUM_BINS_YAW]                 # [T,N,K_yaw]
    logits_ax  = logits_acc[..., NUM_BINS_YAW:]                 # [T,N,K_ax]

    # Stable softmax & log-softmax
    logp_yaw = torch.log_softmax(logits_yaw, dim=-1)            # [T,N,K_yaw]
    p_yaw    = torch.exp(logp_yaw)
    logp_ax  = torch.log_softmax(logits_ax, dim=-1)
    p_ax     = torch.exp(logp_ax)

    # Entropy H(p) = -Σ p log p  (nats); normalized by log(K) to be in [0,1]
    H_yaw = -(p_yaw * logp_yaw).sum(dim=-1)                     # [T,N]
    H_ax  = -(p_ax  * logp_ax ).sum(dim=-1)                     # [T,N]
    Hnorm_yaw = H_yaw / torch.log(torch.tensor(NUM_BINS_YAW, device=logits_acc.device, dtype=logits_acc.dtype))
    Hnorm_ax  = H_ax  / torch.log(torch.tensor(NUM_BINS_AX,  device=logits_acc.device, dtype=logits_acc.dtype))

    # Perplexity = exp(H)
    PPL_yaw = torch.exp(H_yaw)                                   # [T,N]
    PPL_ax  = torch.exp(H_ax)

    # Confidence proxies
    pmax_yaw, _ = p_yaw.max(dim=-1)                              # [T,N]
    pmax_ax,  _ = p_ax.max(dim=-1)

    # Top-2 margin (p1 - p2); larger margin => more confident
    top2_yaw = p_yaw.topk(2, dim=-1).values                      # [T,N,2]
    top2_ax  = p_ax.topk(2, dim=-1).values
    margin_yaw = top2_yaw[..., 0] - top2_yaw[..., 1]             # [T,N]
    margin_ax  = top2_ax[..., 0]  - top2_ax[..., 1]

    # Optional: mean & variance over *bin values* (if you pass bin centers)
    def mean_var_from_bins(p, centers):
        # p: [T,N,K], centers: [K] -> mean/var: [T,N]
        mean = (p * centers.view(1,1,-1)).sum(dim=-1)
        var  = (p * (centers.view(1,1,-1) - mean.unsqueeze(-1))**2).sum(dim=-1)
        return mean, var

    yaw_mean = yaw_var = ax_mean = ax_var = None
    if bin_centers_yaw is not None:
        yaw_mean, yaw_var = mean_var_from_bins(p_yaw, bin_centers_yaw.to(logits_acc))
    if bin_centers_ax is not None:
        ax_mean,  ax_var  = mean_var_from_bins(p_ax,  bin_centers_ax.to(logits_acc))

    # --- Aggregations ---
    # Per-vehicle (average over time)
    per_vehicle = {
        "Hnorm_yaw": Hnorm_yaw.mean(dim=0),     # [N]
        "Hnorm_ax":  Hnorm_ax.mean(dim=0),
        "pmax_yaw":  pmax_yaw.mean(dim=0),
        "pmax_ax":   pmax_ax.mean(dim=0),
        "margin_yaw": margin_yaw.mean(dim=0),
        "margin_ax":  margin_ax.mean(dim=0),
    }
    if yaw_var is not None: per_vehicle |= {"yaw_var": yaw_var.mean(dim=0)}
    if ax_var  is not None: per_vehicle |= {"ax_var":  ax_var.mean(dim=0)}

    # Per-time (average over vehicles) — useful for timeline plots
    per_time = {
        "Hnorm_yaw": Hnorm_yaw.mean(dim=1),     # [T]
        "Hnorm_ax":  Hnorm_ax.mean(dim=1),
        "pmax_yaw":  pmax_yaw.mean(dim=1),
        "pmax_ax":   pmax_ax.mean(dim=1),
        "margin_yaw": margin_yaw.mean(dim=1),
        "margin_ax":  margin_ax.mean(dim=1),
    }

    # Global scalars
    global_avg = {
        "Hnorm_yaw": Hnorm_yaw.mean(),
        "Hnorm_ax":  Hnorm_ax.mean(),
        "pmax_yaw":  pmax_yaw.mean(),
        "pmax_ax":   pmax_ax.mean(),
        "margin_yaw": margin_yaw.mean(),
        "margin_ax":  margin_ax.mean(),
    }

    return {
        "per_tn": {          # per time-step, per vehicle
            "Hnorm_yaw": Hnorm_yaw, "Hnorm_ax": Hnorm_ax,
            "H_yaw": H_yaw, "H_ax": H_ax,
            "PPL_yaw": PPL_yaw, "PPL_ax": PPL_ax,
            "pmax_yaw": pmax_yaw, "pmax_ax": pmax_ax,
            "margin_yaw": margin_yaw, "margin_ax": margin_ax,
            "yaw_mean": yaw_mean, "yaw_var": yaw_var,
            "ax_mean": ax_mean, "ax_var": ax_var,
        },
        "per_vehicle": per_vehicle,
        "per_time": per_time,
        "global_avg": global_avg,
    }

class DynamicLossBalancer:
    """
    Keeps EMAs of loss magnitudes and rescales terms so they contribute
    on a comparable scale. Works with your (term, weight, flag) config.
    """
    def __init__(self, beta: float = 0.98, clamp=(0.25, 4.0), warmup: int = 50, eps: float = 1e-8):
        self.beta = beta
        self.clamp = clamp
        self.warmup = warmup
        self.eps = eps
        self.step = 0
        self.ema = {}  # name(index)-> float

    @torch.no_grad()
    def _update_ema(self, key, value: torch.Tensor):
        v = float(value.detach().mean().cpu())
        if key not in self.ema:
            self.ema[key] = v
        else:
            self.ema[key] = self.beta * self.ema[key] + (1.0 - self.beta) * v

    def __call__(self, cfg, names=None):
        """
        cfg: list of tuples (term_tensor, base_weight, flag_bool)
        names: optional list of unique names matching cfg entries (len==len(cfg)).
               If None, uses integer indices.
        Returns: (balanced_sum, debug_info_dict)
        """
        if names is None:
            names = [f"term_{i}" for i in range(len(cfg))]

        # 1) Update EMAs for enabled terms
        with torch.no_grad():
            for (name, (t, w, f)) in zip(names, cfg):
                if f:
                    self._update_ema(name, t)

        # 2) Compute dynamic scales
        enabled = [(name, t, w) for (name, (t, w, f)) in zip(names, cfg) if f]
        if not enabled:
            # return a safe zero with right device/dtype
            first_term = cfg[0][0]
            return first_term.new_zeros(()), {"scales": {}, "emas": {}}

        # Collect EMA magnitudes for enabled terms
        ema_vals = torch.tensor([max(self.ema.get(name, 1.0), self.eps) for (name, _, _) in enabled],
                                dtype=torch.float32)
        # Target magnitude: mean EMA of enabled terms
        target = float(ema_vals.mean().item())

        scales = {}
        for (name, _, _) in enabled:
            ema_i = max(self.ema.get(name, target), self.eps)
            s = target / ema_i
            # warmup: blend toward 1.0 in the first steps to avoid early oscillation
            if self.step < self.warmup:
                alpha = self.step / max(1, self.warmup)
                s = 1.0 * (1 - alpha) + s * alpha
            # clamp to keep things sane
            s = float(min(max(s, self.clamp[0]), self.clamp[1]))
            scales[name] = s

        # 3) Build final loss
        total = 0.0
        for (name, (t, w, f)) in zip(names, cfg):
            if not f:
                continue
            total = total + (w * scales[name]) * t  # dynamic reweighting

        self.step += 1
        debug = {"scales": scales, "emas": {k: float(v) for k, v in self.ema.items()}}
        return total, debug
