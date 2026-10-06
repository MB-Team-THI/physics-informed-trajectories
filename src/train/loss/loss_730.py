from typing import Tuple, List

from src.train.loss.loss import loss
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
import seaborn as sns

def debug_tensor(t: torch.Tensor, name="tensor", step=None, save_hist=True):
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

def _first_last_valid_indices(val_nt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    val_nt: [n, T] bool
    Returns first_idx [n], last_idx [n] (clamped to [0, T-1])
    """
    n, T = val_nt.shape
    t_ar = torch.arange(T, device=val_nt.device)

    # first true per row (min of t where val, else T)
    first_idx = torch.where(val_nt, t_ar, t_ar.new_full((T,), T)).min(dim=1).values
    # last true per row (max of t where val, else -1)
    last_idx = torch.where(val_nt, t_ar, t_ar.new_full((T,), -1)).max(dim=1).values

    first_idx = first_idx.clamp(0, T - 1)
    last_idx = last_idx.clamp(0, T - 1)
    return first_idx, last_idx

def _displacement_from_first_last(traj_nt2: torch.Tensor,
                                  first_idx_n: torch.Tensor,
                                  last_idx_n: torch.Tensor) -> torch.Tensor:
    """
    traj_nt2: [n, T, 2], first_idx_n/last_idx_n: [n]
    Returns disp_n: [n] Euclidean displacement between first/last valid points.
    """
    n, T, _ = traj_nt2.shape
    fi = first_idx_n.view(n, 1, 1).expand(n, 1, 2)
    li = last_idx_n.view(n, 1, 1).expand(n, 1, 2)
    p_first = torch.gather(traj_nt2, 1, fi).squeeze(1)  # [n,2]
    p_last = torch.gather(traj_nt2, 1, li).squeeze(1)  # [n,2]
    disp = torch.linalg.norm(p_last - p_first, dim=-1)  # [n]
    return disp

def _include_fast_agents(disp_n: torch.Tensor, move_thresh: float) -> torch.Tensor:
    """
    Returns include_n bool where True means agent moved enough.
    """
    return (disp_n > move_thresh)

def _pairwise_mask(val_nt: torch.Tensor, include_n: torch.Tensor) -> torch.Tensor:
    """
    Build [n, n, T] mask:
      - both agents valid at time t
      - both agents included (fast enough)
      - exclude self-pairs
    """
    n, T = val_nt.shape
    device = val_nt.device

    valid_pairs_t = (val_nt.unsqueeze(1) & val_nt.unsqueeze(0))  # [n,n,T]
    inc_pairs = include_n.view(n, 1, 1) & include_n.view(1, n, 1)  # [n,n,1]
    offdiag = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(-1)  # [n,n,1]
    mask = valid_pairs_t & inc_pairs & offdiag  # [n,n,T]
    return mask

def _pairwise_same_t_distances(traj_nt2: torch.Tensor) -> torch.Tensor:
    """
    traj_nt2: [n,T,2]
    Returns dist_nnt: [n,n,T] of pairwise distances at same time indices.
    """
    # torch.cdist expects [batch, m, d] x [batch, n, d] -> [batch, m, n]
    # We can treat time as batch by permuting to [T, n, 2]
    traj_tn2 = traj_nt2.permute(1, 0, 2)  # [T,n,2]
    dist_tnn = torch.cdist(traj_tn2, traj_tn2, p=2)  # [T,n,n]
    dist_nnt = dist_tnn.permute(1, 2, 0)  # [n,n,T]
    return dist_nnt

def _penalty_per_agent_t(
    dist_nnt: torch.Tensor,
    mask_nnt: torch.Tensor,
    d_min: float = 1.0,                   # “hard” distance, e.g. 5 m
    d_soft: float = 3.0,           # start of gentle penalty
    w_soft: float = 0.5,           # weight for gentle term
    w_hard: float = 1.0,            # weight for hard term
    tau_soft: float = 2.0,          # smoothness (bigger = gentler ramp)
    tau_hard: float = 0.2,          # sharpness near d_min (smaller = steeper)
) -> torch.Tensor:
    """
    Smooth piecewise-style penalty:
      - For distances below d_soft (~30 m): mild, wide penalty (soft guardrail).
      - For distances below d_min (e.g., 5 m): steep penalty (hard guardrail).
    Fully differentiable via softplus; mask applied elementwise.

    Returns: [n, T] penalty per agent per time.
    """
    # Gentle term: activates below d_soft with a broad, shallow ramp
    active = (dist_nnt <= d_soft)

    # soft and hard regions (still smooth internally)
    soft_term = w_soft * F.softplus((d_soft - dist_nnt) / tau_soft) ** 2
    hard_term = w_hard * F.softplus((d_min - dist_nnt) / tau_hard) ** 2

    penalty_ijt = (soft_term + hard_term) * active * mask_nnt.float()  # [n,n,T]
    penalty_per_agent_t = penalty_ijt.sum(dim=1)  # [n,T]
    return penalty_per_agent_t

class loss_730(loss):
    def __init__(self,
                idx = 730,
                name = 'TODO',
                description = 'TODO',
                input_ = 'TODO',
                output = 'TODO',
                ) -> None:
        super().__init__(idx,name,description,input_,output)



    # --- main function ---------------------------------------------------------

    def forward(self, X_pred, Y_pred,  # [S, T] flattened over agents
                x_traj_pred_obj_len,  # iterable of agent counts per scenario, len=B
                valid_mask,  # [S, T] bool
                dataset_dict=None,
                d_min: float = 2.0,
                move_thresh: float = 3.0,
                make_plot: bool = False,):

        device = X_pred.device
        loss_sum = X_pred.new_tensor(0.0)
        used = 0

        start = 0

        for s_idx, n in enumerate(x_traj_pred_obj_len):
            n = int(n)
            if n <= 1:
                start += n
                continue

            idx = torch.arange(start, start + n, device=device)
            start += n

            # [n,T,2] positions and [n,T] validity
            traj_nt2 = torch.stack([X_pred[idx], Y_pred[idx]], dim=-1)  # [n,T,2]
            val_nt = valid_mask[idx].bool()  # [n,T]

            # --- displacement filtering
            first_idx_n, last_idx_n = _first_last_valid_indices(val_nt)
            disp_n = _displacement_from_first_last(traj_nt2, first_idx_n, last_idx_n)
            include_n = _include_fast_agents(disp_n, move_thresh)  # [n] bool


            # --- distances and masks
            dist_nnt = _pairwise_same_t_distances(traj_nt2)  # [n,n,T]
            mask_nnt = _pairwise_mask(val_nt, include_n)  # [n,n,T]

            # --- penalty and reduction
            penalty_per_agent_t = _penalty_per_agent_t(dist_nnt, mask_nnt)  # [n,T]

            # agent participates in any valid pair at any time
            active_n = (mask_nnt.any(dim=1)).any(dim=1)  # [n]
            if active_n.any():
                scen_loss = penalty_per_agent_t[active_n].mean()

                loss_sum = loss_sum + scen_loss
                used += 1

            if make_plot:
                _viz_proximity_scenario(traj_nt2, val_nt, include_n, disp_n, d_min, move_thresh,
                                            penalty_per_agent_t)


        if used == 0:
            return X_pred.new_tensor(0.0)

        return loss_sum / used

def _viz_proximity_scenario(traj, val, include, disp, d_min, move_thresh, penalty_per_agent_t=None):
    """
    traj: [n,T,2] (cpu ok), val: [n,T] bool, include: [n] bool, disp: [n]
    penalty_per_agent_t: [n,T] or None
    """

    with torch.no_grad():
        traj_np = traj.detach().cpu().numpy()
        val_np  = val.detach().cpu().numpy().astype(bool)
        inc_np  = include.detach().cpu().numpy().astype(bool)
        disp_np = disp.detach().cpu().numpy()
        n, T, _ = traj_np.shape

        inc_idx = np.where(inc_np)[0]
        ign_idx = np.where(~inc_np)[0]

        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        ax0, ax1, ax2 = axes

        sns.heatmap(penalty_per_agent_t.detach().cpu().numpy(), ax=ax2)

        # ---------------- Left: included agents' valid segments (plain)
        for a in inc_idx:
            pts_ok = val_np[a]
            if pts_ok.sum() < 2:
                continue
            ax0.plot(traj_np[a, pts_ok, 0], traj_np[a, pts_ok, 1], linewidth=1.8, alpha=0.9)
        ax0.set_aspect("equal", adjustable="box")
        ax0.set_title(f"Included agents: {len(inc_idx)} | Ignored (slow≤{move_thresh} m): {len(ign_idx)}\n"
                      f"Ignored idx: {ign_idx.tolist()}")
        ax0.set_xlabel("X [m]"); ax0.set_ylabel("Y [m]")
        ax0.grid(True, linestyle=":", linewidth=0.5)

        # ---------------- Right: colored trajectories by per-point penalty
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]")
        ax1.grid(True, linestyle=":", linewidth=0.5)

        if penalty_per_agent_t is None or inc_idx.size == 0:
            ax1.set_title("No penalty provided – plotting plain included trajectories")
            for a in inc_idx:
                pts_ok = val_np[a]
                if pts_ok.sum() < 2:
                    continue
                ax1.plot(traj_np[a, pts_ok, 0], traj_np[a, pts_ok, 1], linewidth=1.6, alpha=0.9)
            plt.tight_layout()
            plt.show()
            return fig

        pen_np_full = penalty_per_agent_t.detach().cpu().numpy()
        pen_np = pen_np_full[inc_idx]  # [n_inc, T]

        # Build all segment-wise penalty values to get a global normalization
        all_seg_vals = []
        for i, a in enumerate(inc_idx):
            ok = val_np[a]
            # segments exist where both t and t+1 are valid
            ok_seg = ok[:-1] & ok[1:]
            if ok_seg.sum() == 0:
                continue
            # average penalty across the segment (t,t+1)
            c = 0.5 * (pen_np[i, :-1][ok_seg] + pen_np[i, 1:][ok_seg])
            if c.size:
                all_seg_vals.append(c)
        if len(all_seg_vals) == 0:
            ax1.set_title("No valid segments to color")
            plt.tight_layout()
            plt.show()
            return fig

        all_seg_vals = np.concatenate(all_seg_vals)
        # Robust normalization: guard against degenerate constant penalties
        vmin = float(np.nanmin(all_seg_vals))
        vmax = float(np.nanmax(all_seg_vals))
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax): vmax = 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("cool")

        # Plot each agent with a LineCollection colored by segment penalty
        for i, a in enumerate(inc_idx):
            ok = val_np[a]
            if ok.sum() < 2:
                continue
            x = traj_np[a, :, 0]
            y = traj_np[a, :, 1]
            # segment mask: both endpoints valid
            ok_seg = ok[:-1] & ok[1:]
            if ok_seg.sum() == 0:
                continue
            # build segments [ [x_t,y_t], [x_{t+1},y_{t+1}] ]
            x0 = x[:-1][ok_seg]
            y0 = y[:-1][ok_seg]
            x1 = x[1:][ok_seg]
            y1 = y[1:][ok_seg]
            segs = np.stack([np.stack([x0, y0], axis=1),
                             np.stack([x1, y1], axis=1)], axis=1)  # [nseg, 2, 2]
            # segment colors = avg penalty on (t,t+1)
            c = 0.5 * (pen_np[i, :-1][ok_seg] + pen_np[i, 1:][ok_seg])

            lc = LineCollection(segs, array=c, cmap=cmap, norm=norm, linewidths=2.0, alpha=0.95)
            ax1.add_collection(lc)

        # Optionally show ignored agents in light gray (valid parts only)
        for a in ign_idx:
            ok = val_np[a]
            if ok.sum() < 2:
                continue
            ax1.plot(traj_np[a, ok, 0], traj_np[a, ok, 1], color=(0.7, 0.7, 0.7), linewidth=1.0, alpha=0.6)

        ax1.autoscale()  # fits to added collections
        ax1.set_title(f"Trajectories colored by proximity penalty (d_min={d_min} m)")

        # Shared colorbar for penalties
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # matplotlib quirk
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("penalty")

        plt.tight_layout()
        plt.show()
        return fig
