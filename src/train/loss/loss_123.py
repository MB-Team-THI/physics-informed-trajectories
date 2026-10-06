import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class loss_123(nn.Module):
    """
    Differentiable off-map + border-aware loss (drop-in).
    Keeps your interface and adds a smooth 'border zone' penalty on top of the
    existing off-road penalty, while fixing a few indexing details.

    Two terms per timestep:
      1) Off-road (heavy):   L_off  = -log(p_drive + eps)
      2) Border (soft zone): L_edge = softplus(edge_k * (edge_tau - |p_drive - 0.5|))

    Total per point: w_off * L_off + w_edge * L_edge (masked to valid, in-bounds, and moving agents).
    """

    def __init__(self, blur_kernel: int = 5, eps: float = 1e-6, *args, **kwargs):
        super().__init__()
        self.blur_kernel = max(1, int(blur_kernel) | 1)  # force odd
        self.eps = float(eps)

    def _soften(self, m_bin_bhw: torch.Tensor) -> torch.Tensor:
        """
        Local average to create soft edges; keeps values in [0,1].
        m_bin_bhw: [B,H,W] in {0,1} -> returns [B,1,H,W] float in [0,1]
        """
        m = m_bin_bhw.unsqueeze(1).float()  # [B,1,H,W]
        if self.blur_kernel == 1:
            return m
        pad = self.blur_kernel // 2
        # _soften(): change padding to constant 0 so roadness decays near borders
        return F.avg_pool2d(
            F.pad(m, (pad, pad, pad, pad), mode="constant", value=0.0),
            kernel_size=self.blur_kernel,
            stride=1
        )

    def forward(
            self,
            X_pred: torch.Tensor,  # [S, T] world x (meters), flattened over agents
            Y_pred: torch.Tensor,  # [S, T] world y (meters), flattened over agents
            x_traj_pred_obj_len,  # iterable len=B (agents per scenario)
            valid_mask: torch.Tensor,  # [S, T] bool
            image_bin: torch.Tensor,  # [B, H, W] 1=drivable, 0=off
            center_meter: torch.Tensor,  # [B, 2] (cx, cy) meters
            resolution_pix_per_meter: torch.Tensor,  # [B, 2] (rx, ry) px/m
            reduction: str = "mean",
            make_plot: bool = False,
            move_thresh: float = 5.0,  # EXCLUDE slow vehicles: displacement <= move_thresh

            # --- new knobs for the edge-danger field ---
            danger_radius_m: float = 1.0,  # radius (meters) within which danger decays to 0
            decay: str = "linear",  # "linear" or "quadratic"
    ):
        """
        Step 1 (edge-danger only):
          1) Detect edges on the binary infra map.
          2) Build a per-pixel danger field in [0,1] that is 1 at edges and
             decays to 0 linearly/quadratically within 'danger_radius_m'.
          3) Sample the danger field at trajectory points; loss = mean danger over valid in-bounds points.
        """
        device = X_pred.device

        # Normalize infra tensor shape
        if image_bin.dim() == 4:
            image_bin = image_bin.squeeze(1)
        assert image_bin.dim() == 3, "image_bin should be [B,H,W]"
        B, H, W = image_bin.shape
        S, T = X_pred.shape
        assert valid_mask.shape == (S, T), "valid_mask must be [S,T]"

        loss_sum = X_pred.new_tensor(0.0)
        used = 0
        start = 0

        for b, n_agents in enumerate(x_traj_pred_obj_len):
            n = int(n_agents)
            if n <= 0:
                continue

            idx = torch.arange(start, start + n, device=device)
            start += n

            x = X_pred[idx]  # [n,T]
            y = Y_pred[idx]  # [n,T]
            val = valid_mask[idx].bool()  # [n,T]
            if not val.any():
                continue

            # Per-scene transforms
            cx, cy = center_meter[0], center_meter[1]
            rx, ry = resolution_pix_per_meter[0], resolution_pix_per_meter[1]
            # choose conservative pixel radius using the larger axis scale
            Rpx = int(max(1, math.ceil(danger_radius_m * rx)))


            # --- 1) Edge detection on binary map ---
            # Edge = any 3x3 neighborhood contains both 0 and 1 (max != min)
            m = image_bin[b:b + 1, ...].unsqueeze(1).float()  # [1,1,H,W]
            max_nb = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
            min_nb = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
            edge = (max_nb != min_nb).float().squeeze(0).squeeze(0)  # [H,W], 1 on edges

            traj_nt2 = torch.stack([x, y], dim=-1)  # [n,T,2]
            t_ar = torch.arange(T, device=x.device)
            first_idx = torch.where(val, t_ar, t_ar.new_full((T,), T)).min(dim=1).values
            last_idx = torch.where(val, t_ar, t_ar.new_full((T,), -1)).max(dim=1).values
            fi = first_idx.clamp(0, T - 1).view(-1, 1, 1).expand(-1, 1, 2)
            li = last_idx.clamp(0, T - 1).view(-1, 1, 1).expand(-1, 1, 2)
            p_first = torch.gather(traj_nt2, 1, fi).squeeze(1)  # [n,2]
            p_last = torch.gather(traj_nt2, 1, li).squeeze(1)  # [n,2]
            disp = torch.linalg.norm(p_last - p_first, dim=-1)  # [n]
            include = (disp > move_thresh).view(-1, 1)


            # --- 2) Chamfer-style distance to edges up to Rpx (fast, differentiable wrt sampling) ---
            # Initialize distances: 0 at edges, +inf elsewhere
            dist = torch.full((H, W), float('inf'), device=device)
            dist[edge > 0.5] = 0.0

            # Iteratively propagate minimum distance using 3x3 neighborhood (chessboard metric)
            # Cap iterations at Rpx since penalty is zero beyond that radius.
            for _ in range(Rpx):
                # min over 3x3 neighbors
                neigh_min = -F.max_pool2d(-dist.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze(0).squeeze(0)
                # relax
                dist = torch.minimum(dist, neigh_min + 1.0)

            # Clamp to Rpx and convert to [0,1] danger: 1 at edge, 0 at >= Rpx
            dist = torch.clamp(dist, max=Rpx)
            danger_map = 1.0 - (dist / float(Rpx))
            if decay.lower().startswith("quad"):
                danger_map = danger_map * danger_map
            # [H,W] -> [1,1,H,W] for sampling
            danger_map = danger_map.unsqueeze(0).unsqueeze(0)

            # --- Outside-distance field: penalty grows with distance to nearest drivable pixel ---
            # Use same pixel radius as edge danger (or pick a larger one if you want a wider pull)
            Rpx_off = Rpx

            # Initialize distance: 0 on drivable, +inf elsewhere
            drive = image_bin[b].float()  # [H,W], 1=drivable
            dist_to_drive = torch.full((H, W), float('inf'), device=device)
            dist_to_drive[drive > 0.5] = 0.0

            # Chamfer-style distance propagation up to Rpx_off
            for _ in range(Rpx_off):
                neigh_min = -F.max_pool2d(-dist_to_drive.unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze(0).squeeze(0)
                dist_to_drive = torch.minimum(dist_to_drive, neigh_min + 1.0)

            # Normalize to [0,1] (0 on road, 1 at >= Rpx_off from road)
            dist_to_drive = torch.clamp(dist_to_drive, max=Rpx_off) / float(Rpx_off)

            # Optional shaping: quadratic for stronger pull far outside
            off_field_map = dist_to_drive * dist_to_drive if decay.lower().startswith("quad") else dist_to_drive
            off_field_map = off_field_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

            # --- 3) Sample danger at trajectory points ---
            # World (meters) -> pixel coords
            x_px = (x + cx) * rx  # [n,T]
            y_px = H - (y + cy) * ry  # [n,T]  (origin upper-left)

            # In-bounds mask
            in_bounds = (x_px >= 0) & (x_px <= (W - 1)) & (y_px >= 0) & (y_px <= (H - 1))  # [n,T]
            eff_mask = val & in_bounds
            eff_mask = eff_mask & include  # also exclude slow vehicles

            # grid_sample expects normalized coords in [-1,1] with align_corners=True
            Wm1 = max(W - 1, 1)
            Hm1 = max(H - 1, 1)
            u = 2.0 * (x_px / Wm1) - 1.0
            v = 2.0 * (y_px / Hm1) - 1.0
            grid = torch.stack([u, v], dim=-1).reshape(1, n * T, 1, 2)  # [1,P,1,2]

            danger_pt = F.grid_sample(
                danger_map, grid,
                mode="bilinear",
                padding_mode="zeros",  # out-of-bounds -> 0 danger (already masked anyway)
                align_corners=True
            ).view(n, T)  # [n,T]

            # Neutralize invalid steps so they don't contribute
            danger_pt = torch.where(eff_mask, danger_pt, torch.zeros_like(danger_pt))
            # Sample the outside-distance field at trajectory points
            off_pt = F.grid_sample(
                off_field_map, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True
            ).view(n, T)  # [n,T], 0 on road, increases with distance off-road

            # Combine penalties (keep it simple + bounded)
            off_weight = 1.0  # tune if needed
            total_pen = danger_pt + off_weight * off_pt

            # Replace downstream uses of `danger_pt` with `total_pen`
            # 1) neutralize invalid steps:
            total_pen = torch.where(eff_mask, total_pen, torch.zeros_like(total_pen))

            # 2) pick valid values for loss:
            pv = total_pen[eff_mask]


            if pv.numel() == 0:
                continue

            scen_loss = pv.mean() if reduction == "mean" else pv.sum()
            loss_sum = loss_sum + scen_loss
            used += 1

            # Optional viz
            # --- replace your viz block with this ---
            if make_plot:
                try:
                    with torch.no_grad():
                        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

                        # Left: danger field
                        dm = danger_map.squeeze(0).squeeze(0).detach().cpu().numpy()
                        ax0.imshow(dm, origin="upper", cmap="magma", vmin=0.0, vmax=1.0)
                        x_cpu = x_px.detach().cpu().numpy()
                        y_cpu = y_px.detach().cpu().numpy()
                        m_cpu = eff_mask.detach().cpu().numpy()

                        for i in range(n):
                            mask_i = m_cpu[i]
                            if not mask_i.any():
                                continue
                            ax0.plot(x_cpu[i][mask_i], y_cpu[i][mask_i], linewidth=0.8, alpha=0.5, color="white")
                            ax0.scatter(x_cpu[i][mask_i], y_cpu[i][mask_i], s=6, c="white", alpha=0.6)

                        ax0.set_title(f"Danger field (radius={danger_radius_m} m, {decay})")
                        ax0.invert_yaxis()

                        # Right: per-vehicle penalty coloring
                        # danger_pt is [n,T] in [0,1]
                        pen_cpu = total_pen.detach().cpu().numpy()

                        # Optional faint background for context
                        ax1.imshow(dm, origin="upper", cmap="gray", vmin=0.0, vmax=1.0, alpha=0.25)

                        last_sc = None
                        for i in range(n):
                            mask_i = m_cpu[i]
                            if not mask_i.any():
                                continue
                            xi = x_cpu[i][mask_i]
                            yi = y_cpu[i][mask_i]
                            ci = pen_cpu[i][mask_i]  # penalty at each point on this trajectory

                            # light path line + colored points by penalty
                            ax1.plot(xi, yi, linewidth=0.6, alpha=0.4, color="white")
                            last_sc = ax1.scatter(xi, yi, s=10, c=ci, cmap="cool", vmin=0.0, vmax=1.0, alpha=0.95)

                        ax1.set_title("Per-trajectory penalty (colored by danger)")
                        ax1.invert_yaxis()

                        # Colorbar for the right subplot
                        if last_sc is not None:
                            cbar = fig.colorbar(last_sc, ax=ax1, fraction=0.046, pad=0.04)
                            cbar.set_label("Penalty (0-1)")

                        fig.tight_layout()
                        plt.show()
                except Exception:
                    pass

        if used == 0:
            return X_pred.new_tensor(0.0)

        return (loss_sum / used) if reduction == "mean" else loss_sum
