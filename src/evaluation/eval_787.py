import copy
import itertools
import pickle
import random
from math import pi

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from einops import repeat
from matplotlib.widgets import Slider, RadioButtons, Button
from tqdm import tqdm, trange

from src.evaluation.eval_utils import compute_sampling_metrics

UPSAMPLE_FACTOR = 8
NUM_SAMPLES = 32
NUM_ROLLOUTS = 4
SLOW_VEHICLE_THRESHOLD = 0.0


REFERENCE_SCENARIO =52    #4 # 2 # 10
REFERENCE_VEHICLE = 6    #3 #3 # 3
MARGIN = 2.0
TOP = 200
TOTAL_ITERATIONS = 400
FILTER_BASED_ON_DISTANCE = True


def _rep(data):
    if NUM_SAMPLES == 1:
        return data
    if isinstance(data, torch.Tensor):
        data = repeat(data, 'b ... -> (n b) ... ', n=NUM_SAMPLES)
        return data
    elif isinstance(data, list):
        return data * NUM_SAMPLES




MODES = ["argmax", "topk", "topp", "multinomial"]




class eval_787:
    """
    Interactive trajectory evaluator with include/exclude panel.
    - Left panel shows checkboxes (1..N vehicles). Uncheck to exclude a vehicle.
    - Excluded vehicles are zeroed before the model call.
    - Excluded vehicles are hidden from plotting (toggle with 'h').
    """

    def __init__(self,
                 idx=121,
                 name='Interactive nondet scenario evaluation',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=True,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 num_samples=NUM_ROLLOUTS,
                 init_temperature=1.0,
                 min_movement_distance=SLOW_VEHICLE_THRESHOLD,
                 description='Click to set a nondeterministic goal and run multiple samples'):
        self.idx = idx
        self.name = name
        self.input_ = input_
        self.output = output
        self.description = description

        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model
        self.num_samples = num_samples
        self.temperature = init_temperature
        self.sample_mode = "topp"
        self.min_movement_distance = min_movement_distance
        self.zero_image = False  # legacy toggle (kept for back-compat)
        self.random_image = False
        self.random_image_idx = 0

        self.model = None
        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None
        self.all_samples_data = []

        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0
        self.target_len = 30

        self.fig = None
        self.ax = None
        self.vehicle_slider = None
        self.ax_metrics = None

        self.all_predictions = []

        # NEW: state for include/exclude panel and plotting
        self.include_mask = None       # numpy bool array, length = num_vehicles
        self.hide_excluded = True      # if True, excluded vehicles not drawn
        self.check_buttons = None      # CheckButtons widget
        self.ax_panel = None           # axes for the checkbox panel

    @staticmethod
    def _process_dynamic_model(output, gTruthX, gTruthY, gTruthT,
                               x_traj_pred_obj_len, pres_object_lengths_sum):
        X, Y, T = output['X'], output['Y'], output['T']
        Xr = torch.empty_like(gTruthX)
        Yr = torch.empty_like(gTruthY)
        Tr = torch.empty_like(gTruthT)
        for u in range(gTruthX.shape[0]):
            s = pres_object_lengths_sum[u]
            e = pres_object_lengths_sum[u + 1]
            L = x_traj_pred_obj_len[u]
            Xr[u, :L, :, :] = X[s:e, :].unsqueeze(2)
            Yr[u, :L, :, :] = Y[s:e, :].unsqueeze(2)
            Tr[u, :L, :, :] = T[s:e, :].unsqueeze(2)
        return Xr, Yr, Tr

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        self.model = model
        self.dataloader_test = dataloader_test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_dict = dataset_dict
        return self._evaluate()

    def _get_num_vehicles_in_scenario(self, scenario_sample):
        start_idx = scenario_sample['hist_object_lengths_sum'][0]
        end_idx = scenario_sample['hist_object_lengths_sum'][1]
        return len(scenario_sample['hist_objs_seq_len'][start_idx:end_idx])

    def _plot_matched_predictions(self, matches, reference_scenario, reference_vehicle, norm_trajs):
        """
        Plots world-frame predictions (left) and normalized predictions (right).
        norm_trajs: list of normalized trajectories (each np.array of shape (T,2)),
                    including the reference trajectory as the first element.
        """
        samples = self.all_samples_data

        def _pred_xy(sample, vidx):
            x = sample['pred_objsx'][0, vidx].detach().cpu().numpy().squeeze(-1)
            y = sample['pred_objsy'][0, vidx].detach().cpu().numpy().squeeze(-1)
            T = int(sample['pred_objs_seq_len'][vidx]) if 'pred_objs_seq_len' in sample else len(x)
            return np.stack([x[:T], y[:T]], axis=1)


        # Figure: left = world frame, right = normalized
        fig, (ax_world, ax_norm) = plt.subplots(1, 2, figsize=(12, 6))

        # --- world frame ---
        ref_sample = samples[reference_scenario]
        ref_xy = _pred_xy(ref_sample, reference_vehicle)
        if ref_xy is None:
            raise ValueError("Reference prediction not found in sample.")

        ax_world.plot(ref_xy[:, 0], ref_xy[:, 1], 'k-', lw=2, label='ref pred')
        ax_world.scatter(ref_xy[-1, 0], ref_xy[-1, 1], c='k', marker='x', s=80)

        for (si, vi) in matches:
            xy = _pred_xy(samples[si], vi)
            if xy is None:
                continue
            ax_world.plot(xy[:, 0], xy[:, 1], lw=1, alpha=0.8)
            ax_world.scatter(xy[-1, 0], xy[-1, 1], s=20, alpha=0.8)

        ax_world.set_aspect('equal')
        ax_world.set_title("World-frame predictions")
        ax_world.legend()

        # --- normalized trajectories ---
        for i, traj in enumerate(norm_trajs):
            if i == 0:
                ax_norm.plot(traj[:, 0], traj[:, 1], 'k-', lw=2, label='reference')
                ax_norm.scatter(traj[-1, 0], traj[-1, 1], c='k', marker='x', s=80)
            else:
                ax_norm.plot(traj[:, 0], traj[:, 1], lw=1, alpha=0.8)
                ax_norm.scatter(traj[-1, 0], traj[-1, 1], s=20, alpha=0.8)

        ax_norm.axhline(0, lw=0.5, color='gray')
        ax_norm.axvline(0, lw=0.5, color='gray')
        ax_norm.set_aspect('equal')
        ax_norm.set_title("Normalized predictions (start at 0,0)")
        ax_norm.legend()
        # ax_norm.set_ylim(-2,2)
        plt.tight_layout()
        plt.show()

    def _find_trajectories_with_similiar_trajectory(self, top_k,reference_scenario,reference_vehicle ):
        samples = self.all_samples_data
        margin = MARGIN


        def _last_xy(sample, vidx):
            xy = sample['hist_objs'][0, vidx].cpu().numpy()  # (T, D)
            return xy[sample['hist_objs_seq_len'][vidx] - 1, :2] if 'hist_objs_seq_len' in sample else xy[-1, :2]

        def _goal_xy(sample, vidx):
            return sample['cond_goal_point'][0, vidx].cpu().numpy()  # (2,)

        def _last_yaw(sample, vidx):
            psi_seq = sample['hist_objs'][0, vidx][:,3] # (T,)
            # psi_seq = sample['hist_objs'][0, vidx].cpu().numpy().squeeze(-1)  # (T,)
            t = -1
            return float(psi_seq[t])


        def _to_local_points(points_xy, origin_xy, yaw):
            # Translate to origin, then rotate by -yaw so heading aligns with +x
            pts = points_xy - origin_xy[None, :]
            c, s = np.cos(-yaw), np.sin(-yaw)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            return (pts @ R.T).astype(np.float32)

        def _pred_xy(sample, vidx):
            # Fallback: split x/y
            x = sample['pred_objsx'][0, vidx].detach().cpu().numpy().squeeze(-1)
            y = sample['pred_objsy'][0, vidx].detach().cpu().numpy().squeeze(-1)
            T = int(sample['pred_objs_seq_len'][vidx]) if 'pred_objs_seq_len' in sample else min(len(x), len(y))
            return np.stack([x[:T], y[:T]], axis=1)

        # --- reference descriptor for matching (goal displacement in local frame) ---
        ref_sample = samples[reference_scenario]
        ref_pos = _last_xy(ref_sample, reference_vehicle)
        ref_goal = _goal_xy(ref_sample, reference_vehicle)
        ref_yaw = _last_yaw(ref_sample, reference_vehicle)
        ref_delta_local = _to_local_points(ref_goal[None, :], ref_pos, ref_yaw)[0]

        # Also compute the normalized reference prediction (so you can compare shapes)
        ref_pred = _pred_xy(ref_sample, reference_vehicle)
        ref_norm_traj = None
        if ref_pred is not None:
            ref_norm_traj = _to_local_points(ref_pred, ref_pos, ref_yaw)  # starts at (0,0), heading +x

        # --- find matches (by local goal displacement within margin) ---
        T_ref = ref_norm_traj.shape[0]

        matches = []
        for si, sample in tqdm(enumerate(samples)):
            n_vehicles = self._get_num_vehicles_in_scenario(sample)
            for vi in range(n_vehicles):
                try:
                    pred_xy = _pred_xy(sample, vi)
                    pos = _last_xy(sample, vi)
                    yaw = _last_yaw(sample, vi)
                except Exception:
                    continue

                # normalize candidate traj (start at 0,0; heading +x)
                norm_xy = _to_local_points(pred_xy, pos, yaw)

                # mean per-step Euclidean distance vs. reference (align lengths)
                L = min(T_ref, norm_xy.shape[0])
                if norm_xy.shape[0] < 30:
                    continue
                traj_dist = float(np.linalg.norm(norm_xy[:L] - ref_norm_traj[:L], axis=1).mean())
                last_step_dist = float(np.linalg.norm(norm_xy[L - 1] - ref_norm_traj[L - 1]))

                if last_step_dist <= margin:
                    matches.append((si, vi, traj_dist, norm_xy))

        if FILTER_BASED_ON_DISTANCE:
            matches.sort(key=lambda x: x[2])
        matches = matches[:top_k]
        ret = []
        norm_trajs = []

        # Build list of normalized predicted trajectories aligned to same start/orientation
        # matches.append((reference_scenario, reference_vehicle, 0.0))
        for si, vi, dist, norm_xy  in tqdm(matches):
            pred_xy = _pred_xy(samples[si], vi)
            if pred_xy is None:
                continue  # skip if no prediction available
            pos = _last_xy(samples[si], vi)
            yaw = _last_yaw(samples[si], vi)
            norm_xy = _to_local_points(pred_xy, pos, yaw)  # start at (0,0), heading +x
            ret.append((si, vi))
            norm_trajs.append(norm_xy)

        # Optional: quick debug plot (comment out if not needed)
        # self._plot_matched_predictions(ret, reference_scenario, reference_vehicle, norm_trajs)
        matches = [(m[0],m[1], m[2]) for m in matches]
        with open("matches.pkl", "wb") as f:
            pickle.dump(matches, f)
        norm_trajs = np.stack(norm_trajs, axis=0)
        np.save("norm_trajs.npy", norm_trajs)


        return ret, norm_trajs

    @staticmethod
    def compute_2d_rms_spread(norm_trajs):
        """
        Compute the 2D RMS spread across normalized trajectories.

        Args:
            norm_trajs: list of np.ndarray, each of shape (T, 2),
                        aligned so that all start at (0,0) and face +x.

        Returns:
            spread_per_t: np.ndarray of shape (T,), RMS spread per timestep [m].
            mean_spread: float, average spread across time.
        """
        # stack into [N, T, 2]
        # filter out trajs that are too short
        norm_trajs = [traj for traj in norm_trajs if traj.shape[0] == 30]
        trajs = np.stack(norm_trajs, axis=0)
        N, T, _ = trajs.shape

        # mean trajectory [T, 2]
        mu = np.mean(trajs, axis=0)

        # deviations [N, T, 2]
        diffs = trajs - mu[None, :, :]

        # 2D RMS spread per time step
        spread_per_t = np.sqrt(np.mean(np.sum(diffs ** 2, axis=2), axis=0))  # [T]

        # mean spread over horizon
        mean_spread = float(np.mean(spread_per_t))

        return spread_per_t, mean_spread

    def _evaluate(self):
        from matplotlib.widgets import Slider, RadioButtons, CheckButtons  # ensure available
        self.model.eval()
        self.model.to(self.device)
        self.all_samples_data = []
        for i, sample in enumerate(tqdm(self.dataloader_test(epoch=0), desc="Loading data")):
            self.all_samples_data.append(sample)
            if i >= TOTAL_ITERATIONS:
                break
        while 1:

            try:
                scenario_idx = int(input(f"Select reference scenario index\t"))
                vehicle_idx = int(input(f"Select reference vehicle index\t"))
                _, norm_traj = self._find_trajectories_with_similiar_trajectory(top_k=TOP, reference_scenario=scenario_idx, reference_vehicle=vehicle_idx)
                print(f"Found {len(norm_traj)} similar trajectories.")
                for i in range(10, len(norm_traj), 10):
                    spread_per_t, mean_spread = self.compute_2d_rms_spread(norm_traj[:i])

                    print(f"2D RMS spread TOP {i}: {mean_spread:.2f} m")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue

        # pick scenario
        s = "5"
        self.current_scenario_idx = max(0, min(len(self.all_samples_data) - 1, int(s) - 1)) if s else 0
        self.original_sample_cache = self.all_samples_data[self.current_scenario_idx]
        self.sample_cache = self.all_samples_data[self.current_scenario_idx]

        # pick active vehicle
        v = "4"
        num_vehicles = self._get_num_vehicles_in_scenario(self.sample_cache)
        self.current_vehicle_idx = max(0, min(num_vehicles - 1, int(v) - 1)) if v else 0

        # NEW: initialize include mask (all included)
        import numpy as np
        self.include_mask = np.ones(num_vehicles, dtype=bool)


        def _compute_metrics_text():
            """
            Metrics panel:
              - For each *kept* vehicle, compute per-timestep 2D spread across rollouts:
                    std2d_t = sqrt( Var[x_t] + Var[y_t] )
                then average over valid timesteps (vehicle-specific length).
              - Labels use ORIGINAL indices (1-based) even though outputs are compacted.
            """
            header = "Metrics Panel\nPer-timestep 2D std, averaged over time"

            # guards
            if self.include_mask is None:
                return header + "\n\n(no vehicles)"
            keep = np.flatnonzero(self.include_mask)  # kept -> original mapping
            if keep.size == 0:
                return header + "\n\n(no vehicles included)"
            if not self.all_predictions:
                return header + "\n\n(no predictions yet)"

            # Stack rollouts: xs, ys -> [S, N_kept, T]
            xs_list, ys_list = [], []
            for run in self.all_predictions:
                X = run['X_reshaped'][0]  # [N_kept, T, 1] or [N_kept, T]
                Y = run['Y_reshaped'][0]
                X = X.detach().cpu().numpy()
                Y = Y.detach().cpu().numpy()
                X = X[:, :, 0]
                Y = Y[:, :, 0]
                xs_list.append(X)  # [N_kept, T]
                ys_list.append(Y)  # [N_kept, T]

            xs = np.stack(xs_list, axis=0)  # [S, N_kept, T]
            ys = np.stack(ys_list, axis=0)  # [S, N_kept, T]
            S, N_kept, T = xs.shape

            # Get per-vehicle valid prediction lengths after masking (avoid padding bias)
            # We reconstruct the compacted sample and read its pred seq lens.
            sc = clone_torch_tree(self.original_sample_cache, device="cpu")
            sc = remove_ingored_vehicles(sc, self.include_mask, device="cpu")
            try:
                lengths = sc['pred_objs_seq_len'][sc['pres_object_lengths_sum'][0]:sc['pres_object_lengths_sum'][1]]
            except KeyError:
                # Fallback: if lengths are not available, assume full T
                lengths = [T] * N_kept


            # Chi-square quantiles for 2D confidence ellipses:
            # Ellipse defined by (z - μ)^T Σ^{-1} (z - μ) <= χ^2_{2,α}
            # Area(α) = π * χ^2_{2,α} * sqrt(det(Σ))
            CHI2_50 = 1.3862943611198906  # ≈ χ^2 with dof=2 at α=0.50
            CHI2_90 = 4.605170185988092  # ≈ χ^2 with dof=2 at α=0.90

            def _safe_det2x2(a11, a12, a21, a22):
                """Determinant of 2x2 matrix [[a11,a12],[a21,a22]] with NaN safety."""
                return a11 * a22 - a12 * a21

            def _eig2x2_sym(a11, a12, a22):
                """
                Eigenvalues of symmetric 2x2 [[a11, a12], [a12, a22]].
                Returns λ1 >= λ2 (non-negative for covariance).
                """
                trace = a11 + a22
                diff = a11 - a22
                disc = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * a12 * a12))
                lam1 = 0.5 * (trace + disc)
                lam2 = 0.5 * (trace - disc)
                return lam1, lam2

            # xs, ys: expected to be arrays of shape [K, N_kept, T]
            # lengths: per-vehicle effective horizon (<= T), optional
            # keep: indices mapping kept vehicles to ORIGINAL ids
            # N_kept, T should already be defined in the surrounding code

            per_vehicle_avg_std = np.zeros(N_kept, dtype=np.float32)  # (legacy metric) avg sqrt(var_x + var_y)
            per_vehicle_area50_mean = np.zeros(N_kept, dtype=np.float32)  # time-avg area of 50% ellipse
            per_vehicle_area90_mean = np.zeros(N_kept, dtype=np.float32)  # time-avg area of 90% ellipse
            per_vehicle_endpoint_std_major = np.zeros(N_kept, dtype=np.float32)  # √λ_max at endpoint (principal std)
            per_vehicle_endpoint_std_minor = np.zeros(N_kept, dtype=np.float32)  # √λ_min at endpoint
            per_vehicle_endpoint_area90 = np.zeros(N_kept, dtype=np.float32)  # 90% ellipse area at endpoint

            # ---- Core loop: compute per-vehicle, per-time covariance and summarize ----
            # Explanation of what we compute:
            #  - For each time step t, from K samples we estimate:
            #      μ_t = mean([x_t, y_t]) over K
            #      Σ_t = covariance([x_t, y_t]) over K
            #  - We then derive:
            #      • 2D RMS spread: sqrt(trace(Σ_t)) = sqrt(var_x + var_y)
            #      • Ellipse areas for 50% and 90%: π * χ^2_(2,α) * sqrt(det(Σ_t))
            #      • Endpoint principal stds: sqrt(λ_max), sqrt(λ_min) from Σ_T
            #  - Finally, we average these across time (except endpoint stats).

            for v in range(N_kept):
                # Effective length for vehicle v (guards against variable horizons)
                L = int(lengths[v]) if v < len(lengths) else T
                L = max(1, min(L, T))

                # Extract KxL matrices for this vehicle
                X = xs[:, v, :L]  # shape [K, L]
                Y = ys[:, v, :L]  # shape [K, L]

                # Mean over K (per time step)
                mu_x = np.nanmean(X, axis=0)  # [L]
                mu_y = np.nanmean(Y, axis=0)  # [L]

                # Centered variables
                Xc = X - mu_x[None, :]  # [K, L]
                Yc = Y - mu_y[None, :]  # [K, L]

                # Unbiased covariance estimates (divide by K-1). We compute the 2x2 Σ_t entries explicitly.
                denom = max(1, X.shape[0] - 1)  # guard if K==1
                # Variances
                var_x_t = np.nansum(Xc * Xc, axis=0) / denom  # [L]
                var_y_t = np.nansum(Yc * Yc, axis=0) / denom  # [L]
                # Covariance (symmetric)
                cov_xy_t = np.nansum(Xc * Yc, axis=0) / denom  # [L]

                # ---- Time-marginal scalars ----
                # 2D RMS spread per t: sqrt( tr(Σ_t) ) = sqrt(var_x + var_y)
                std2d_t = np.sqrt(np.maximum(0.0, var_x_t + var_y_t))  # [L]

                # Determinant per t for ellipse areas
                det_t = _safe_det2x2(var_x_t, cov_xy_t, cov_xy_t, var_y_t)  # [L]
                det_t = np.maximum(det_t, 0.0)  # numerical guard

                # Ellipse areas at 50% / 90% confidence (per t)
                # Area(α,t) = π * χ^2_(2,α) * sqrt(det(Σ_t))
                area50_t = pi * CHI2_50 * np.sqrt(det_t)  # [L]
                area90_t = pi * CHI2_90 * np.sqrt(det_t)  # [L]

                # ---- Endpoint stats (at last valid step) ----
                t_end = L - 1
                # Eigenvalues of Σ_T to get principal stds (major/minor axes)
                lam1, lam2 = _eig2x2_sym(var_x_t[t_end], cov_xy_t[t_end], var_y_t[t_end])
                std_major = np.sqrt(max(0.0, lam1))
                std_minor = np.sqrt(max(0.0, lam2))
                det_end = max(0.0, _safe_det2x2(var_x_t[t_end], cov_xy_t[t_end], cov_xy_t[t_end], var_y_t[t_end]))
                area90_end = pi * CHI2_90 * np.sqrt(det_end)

                # ---- Aggregate time-marginal summaries ----
                per_vehicle_avg_std[v] = float(np.nanmean(std2d_t))
                per_vehicle_area50_mean[v] = float(np.nanmean(area50_t))
                per_vehicle_area90_mean[v] = float(np.nanmean(area90_t))
                per_vehicle_endpoint_std_major[v] = float(std_major)
                per_vehicle_endpoint_std_minor[v] = float(std_minor)
                per_vehicle_endpoint_area90[v] = float(area90_end)

            # ---- Build human-readable lines, mapped back to ORIGINAL ids (1-based) ----
            lines = []
            for k, orig in enumerate(keep):
                vid = int(orig) + 1
                lines.append(f"Veh {vid}:")
                lines.append(f"Avg 2D spread: {per_vehicle_avg_std[k]:.2f} m")
                # lines.append(f"  Mean ellipse area 50%:          {per_vehicle_area50_mean[k]:.2f} m^2")
                # lines.append(f"  Mean ellipse area 90%:          {per_vehicle_area90_mean[k]:.2f} m^2")
                lines.append(
                    f"FD major/minor std:       {per_vehicle_endpoint_std_major[k]:.2f} m / {per_vehicle_endpoint_std_minor[k]:.2f} m")
                # lines.append(f"  Endpoint area 90%:              {per_vehicle_endpoint_area90[k]:.2f} m^2")
                lines.append("")  # blank line between vehicles

            # ---- Optional overall summaries (choose a representative scalar; here: avg 2D spread) ----
            if N_kept > 0:
                overall_mean_spread = float(np.nanmean(per_vehicle_avg_std))
                overall_max_spread = float(np.nanmax(per_vehicle_avg_std))
            else:
                overall_mean_spread = np.nan
                overall_max_spread = np.nan

            lines.append(f"Overall mean 2D spread: {overall_mean_spread:.2f} m")
            lines.append(f"Overall max  2D spread: {overall_max_spread:.2f} m")

            # If your UI expects a single string:
            metrics_text = "\n".join(lines)
            return header + "\n\n" + metrics_text

        def _render_metrics_panel():
            """Ensures the panel exists and renders the metrics text."""
            if self.ax_metrics is None:
                # [left, bottom, width, height] in figure coords
                self.ax_metrics = plt.axes([0.80, 0.22, 0.18, 0.70])
                self.ax_metrics.set_facecolor((0.08, 0.08, 0.08))

            self.ax_metrics.clear()
            self.ax_metrics.axis('off')
            txt = _compute_metrics_text()
            self.ax_metrics.text(
                0.02, 0.98, txt,
                transform=self.ax_metrics.transAxes,
                va='top', ha='left',
                family='monospace', fontsize=10
            )

        def clone_torch_tree(x, *, detach=True, device=None):
            """Recursively clone a pytree of dict/list/tuple/tensors.
            detach=True drops autograd history. device can move tensors."""
            if torch.is_tensor(x):
                t = x.detach().clone() if detach else x.clone()
                return t.to(device) if device is not None else t
            elif isinstance(x, dict):
                return {k: clone_torch_tree(v, detach=detach, device=device) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                seq = [clone_torch_tree(v, detach=detach, device=device) for v in x]
                return type(x)(seq)
            else:
                # for numbers/strings -> as-is; for numpy/others -> deepcopy to be safe
                return copy.deepcopy(x)

        def remove_ingored_vehicles(sc, include_mask, device=None) -> torch.Tensor:
            if device is not None:
                sc = clone_torch_tree(sc,device=device)
            else:
                sc = clone_torch_tree(sc,device=self.device)
            sc["hist_objs"] = sc["hist_objs"][:,include_mask]
            sc["obj_decoder_in"] = sc["obj_decoder_in"][:,include_mask]
            sc["pred_objsx"] = sc["pred_objsx"][:,include_mask]
            sc["pred_objsy"] = sc["pred_objsy"][:,include_mask]
            sc["pred_objst"] = sc["pred_objst"][:,include_mask]
            sc["hist_objs_seq_len"] = [x for x, m in zip(sc["hist_objs_seq_len"], include_mask) if m]
            sc["hist_object_lengths_sum"] = [0, int(include_mask.sum())]
            sc["hist_obj_lens"] =[int(include_mask.sum())]
            sc["pred_obj_lens"] =[int(include_mask.sum())]
            sc["cond_goal_point"] = sc["cond_goal_point"][:, include_mask]
            return sc

        def run_model():
            if all(~self.include_mask):
                print("No vehicles included; skipping model run.")
                return []
            sc = clone_torch_tree(self.original_sample_cache)
            sc = remove_ingored_vehicles(sc, self.include_mask)
            self.sample_cache = sc

            x_image = sc['images'].to(self.device)
            if self.zero_image:
                # zero-out the image input while keeping shape/device/dtype
                x_image = torch.zeros_like(x_image)
            elif self.random_image:
                print(f"Using random image for x_image input. {self.random_image_idx}")
                x_image = self.all_samples_data[self.random_image_idx]['images'].to(self.device)

            cond = torch.cat([sc['cond_goal_point'].to(self.device).float()], dim=2)
            x_traj_len = sc['hist_objs_seq_len']
            x_traj = [sc['hist_objs'].to(self.device), sc['hist_obj_lens']]
            batch_wise_decoder_input = sc['obj_decoder_in'].to(self.device)

            # replicate for sampling
            x_image = _rep(x_image)
            x_traj = [_rep(x_traj[0]), _rep(x_traj[1])]
            x_traj_len = _rep(x_traj_len)
            conditions = _rep(cond)
            batch_wise_decoder_input = _rep(batch_wise_decoder_input)


            Ns = self._get_num_vehicles_in_scenario(sc)
            batch_wise_object_lengths_sum = torch.arange(NUM_SAMPLES + 1, device=self.device, dtype=torch.long) * Ns

            with torch.no_grad():
                out = self.model(
                    x_image=x_image, x_traj=x_traj, x_traj_len=x_traj_len,
                    batch_wise_object_lengths_sum=batch_wise_object_lengths_sum,
                    conditions=conditions, batch_wise_decoder_input=batch_wise_decoder_input,
                    target_length=self.target_len, temperature=self.temperature, sample_mode=self.sample_mode
                )

            Xr, Yr, _ = self._process_dynamic_model(
                out,
                _rep(sc['pred_objsx'].to(self.device)),
                _rep(sc['pred_objsy'].to(self.device)),
                _rep(sc['pred_objst'].to(self.device)),
                sc['pred_obj_lens'] * NUM_SAMPLES,
                batch_wise_object_lengths_sum
            )

            runs = [{'X_reshaped': Xr[k:k + 1], 'Y_reshaped': Yr[k:k + 1]} for k in range(NUM_SAMPLES)]
            return runs

        def plot_all():
            if self.fig is None:
                self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 12))

            img = self._get_background_image()
            self.ax.clear()
            self.ax.imshow(img)
            self.ax.axis('off')

            sc = clone_torch_tree(self.original_sample_cache)
            sc = remove_ingored_vehicles(sc, self.include_mask)
            kept = np.flatnonzero(self.include_mask)
            self.sample_cache = sc
            num_vehicles = self._get_num_vehicles_in_scenario(sc)

            # filter by movement + include_mask
            moving_vehicle_indices = []
            traj_hist = sc['hist_objs'][0].cpu().numpy()
            lengths = sc['hist_objs_seq_len'][sc['hist_object_lengths_sum'][0]:sc['hist_object_lengths_sum'][1]]

            for v_idx in range(num_vehicles):
                L = lengths[v_idx]
                if L <= 1:
                    continue
                distance = np.linalg.norm(traj_hist[v_idx, 0, :2] - traj_hist[v_idx, L - 1, :2])
                if distance > self.min_movement_distance:
                    moving_vehicle_indices.append(v_idx)



            # draw
            cmap = plt.get_cmap('tab20')
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(
                self.dataset_dict[0]['bbox_meter']) * UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR
            _ = compute_sampling_metrics(self.all_predictions)  # rf not used here
            # orig2comp = {int(o): i for i, o in enumerate(kept)}
            # moving_vehicle_indices = [int(i) for i in moving_vehicle_indices if int(i) in orig2comp]

            for k in moving_vehicle_indices:
                vehicle_color = cmap(k  % cmap.N)


                self._draw_history(img, color=(.6, .6, .6), vehicle_idx=k, include_mask=self.include_mask)
                if self.all_predictions:
                    pink_color = (1.0, 0.41, 0.7)
                    for run in self.all_predictions:
                        self._draw_run(img, run, color=vehicle_color, goal_point_color=vehicle_color, vehicle_idx=k)

                # vehicle label
                L = lengths[v_idx]
                last_hist_point_world = traj_hist[v_idx, L - 1, :2]
                pix = ((last_hist_point_world + center) * res).astype(int)
                pix[1] = H - pix[1]
                # print(f"place 1 for vehicle {v_idx + 1}")
                self.ax.text(pix[0] + 10, pix[1], f"{v_idx + 1}",
                             color='white',
                             backgroundcolor=vehicle_color,
                             fontsize=10, fontweight='bold',
                             ha='left', va='center',
                             bbox=dict(boxstyle="round,pad=0.2", fc=vehicle_color, ec='none'))

            #### ONLY TEXT


            sc = self.original_sample_cache
            num_vehicles = self._get_num_vehicles_in_scenario(sc)

            # filter by movement + include_mask
            moving_vehicle_indices = []
            traj_hist = sc['hist_objs'][0].cpu().numpy()
            lengths = sc['hist_objs_seq_len'][sc['hist_object_lengths_sum'][0]:sc['hist_object_lengths_sum'][1]]

            for v_idx in range(num_vehicles):
                L = lengths[v_idx]
                if L <= 1:
                    continue
                distance = np.linalg.norm(traj_hist[v_idx, 0, :2] - traj_hist[v_idx, L - 1, :2])
                if distance > self.min_movement_distance:
                    moving_vehicle_indices.append(v_idx)

            print(f"Plotting {len(moving_vehicle_indices)} of {num_vehicles} vehicles "
                  f"(moved > {self.min_movement_distance}m, hidden={self.hide_excluded}).")

            # draw
            cmap = plt.get_cmap('tab20')
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(
                self.dataset_dict[0]['bbox_meter']) * UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR
            _ = compute_sampling_metrics(self.all_predictions)  # rf not used here

            for v_idx in moving_vehicle_indices:
                if not self.include_mask[v_idx]:
                    continue
                vehicle_color = cmap(v_idx % cmap.N)

                # vehicle label
                L = lengths[v_idx]
                last_hist_point_world = traj_hist[v_idx, L - 1, :2]
                pix = ((last_hist_point_world + center) * res).astype(int)
                pix[1] = H - pix[1]
                print(f"place 2 for vehicle {v_idx + 1}")
                self.ax.text(pix[0] + 10, pix[1], f"{v_idx + 1}",
                             color='white',
                             backgroundcolor=vehicle_color,
                             fontsize=10, fontweight='bold',
                             ha='left', va='center',
                             bbox=dict(boxstyle="round,pad=0.2", fc=vehicle_color, ec='none'))
            _render_metrics_panel()
            self.ax.imshow(img)
            self.fig.canvas.draw_idle()

        def get_bg():
            sc = self.sample_cache
            img_gray = sc['images'][0, 0].cpu().numpy()
            if self.zero_image:
                img_gray = np.zeros_like(img_gray)
            elif self.random_image:
                img_gray = self.all_samples_data[self.random_image_idx]['images'][0, 0].cpu().numpy()
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            return cv2.resize(img_rgb, None, fx=UPSAMPLE_FACTOR, fy=UPSAMPLE_FACTOR,
                              interpolation=cv2.INTER_LINEAR)

        self._get_background_image = get_bg

        def draw_hist(img, color, vehicle_idx, include_mask):
            # skip excluded if hidden
            sc = clone_torch_tree(self.original_sample_cache)
            sc = remove_ingored_vehicles(sc, self.include_mask)
            self.sample_cache = sc
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(
                self.dataset_dict[0]['bbox_meter']) * UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR
            traj = sc['hist_objs'][0].cpu().numpy()
            lengths = sc['hist_objs_seq_len'][sc['hist_object_lengths_sum'][0]:sc['hist_object_lengths_sum'][1]]

            if vehicle_idx >= len(lengths):
                return
            L = lengths[vehicle_idx]
            pts = traj[vehicle_idx, :L, :2]
            pix = ((pts + center) * res).astype(int)
            pix[:, 1] = H - pix[:, 1]
            cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, color, thickness=2 * UPSAMPLE_FACTOR)

        self._draw_history = draw_hist

        def draw_run(img, run, color, goal_point_color, vehicle_idx):
            # skip excluded if hidden
            sc = clone_torch_tree(self.original_sample_cache,device="cpu")
            sc = remove_ingored_vehicles(sc, self.include_mask, device="cpu")
            self.sample_cache = sc
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(
                self.dataset_dict[0]['bbox_meter']) * UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR

            Xp = run['X_reshaped'][0].cpu().numpy()[:, :, 0]
            Yp = run['Y_reshaped'][0].cpu().numpy()[:, :, 0]
            lengths = sc['pred_objs_seq_len'][sc['pres_object_lengths_sum'][0]:sc['pres_object_lengths_sum'][1]]

            if vehicle_idx >= len(lengths):
                assert False
                return
            L = lengths[vehicle_idx]
            pts = np.stack([Xp[vehicle_idx, :L], Yp[vehicle_idx, :L]], axis=-1)
            pix = ((pts + center) * res).astype(int)
            pix[:, 1] = H - pix[:, 1]
            cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, color, thickness=4)

            gx, gy = sc['cond_goal_point'][0, vehicle_idx]
            pixg = (((np.array([gx, gy]) + center) * res)).astype(int)
            pixg[1] = H - pixg[1]
            cv2.drawMarker(img, tuple(pixg), goal_point_color, markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=8 * UPSAMPLE_FACTOR, thickness=2 * UPSAMPLE_FACTOR)

        self._draw_run = draw_run

        def on_click(event):
            if event.inaxes != self.ax:
                return
            Wp, Hp = self.dataset_dict[0]['bbox_pixel']
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(
                self.dataset_dict[0]['bbox_meter']) * UPSAMPLE_FACTOR
            Hp *= UPSAMPLE_FACTOR

            xw = event.xdata / res[0] - center[0]
            yw = (Hp - event.ydata) / res[1] - center[1]
            print(f"Setting new goal for ACTIVE vehicle {self.current_vehicle_idx + 1}: ({xw:.2f}, {yw:.2f})")

            self.original_sample_cache['cond_goal_point'][0, self.current_vehicle_idx, 0] = xw
            self.original_sample_cache['cond_goal_point'][0, self.current_vehicle_idx, 1] = yw

            self.all_predictions.clear()
            for _ in trange(self.num_samples, desc="Sampling trajectories"):
                self.all_predictions.extend(run_model())
            plot_all()

        def on_key(event):
            if event.key == 'r':
                print("Resetting predictions.")
                self.all_predictions.clear()
                plot_all()
            elif event.key == 'h':
                self.hide_excluded = not self.hide_excluded
                print(f"hide_excluded = {self.hide_excluded}")
                plot_all()

        # initial draw
        plot_all()

        # layout: leave room for bottom controls; left panel is built below
        plt.subplots_adjust(left=0.16, bottom=0.2)
        plt.subplots_adjust(right=0.78)  # leave space for the metrics panel

        # sliders + radios
        ax_temp = plt.axes([0.25, 0.1, 0.50, 0.03])
        temp_slider = Slider(ax=ax_temp, label='Temperature', valmin=0.0, valmax=20.0, valinit=self.temperature,
                             valstep=0.1)

        ax_scenario = plt.axes([0.1, 0.04, 0.35, 0.03])
        scenario_slider = Slider(ax=ax_scenario, label='Scenario', valmin=1, valmax=len(self.all_samples_data),
                                 valinit=self.current_scenario_idx + 1, valstep=1)

        # --- Scenario +/- buttons (simple nudge around the slider) ---
        ax_prev = plt.axes([0.06, 0.04, 0.03, 0.03])  # small square to the left of the slider
        ax_next = plt.axes([0.50, 0.04, 0.03, 0.03])  # small square to the right of the slider
        btn_prev = Button(ax_prev, '◀')
        btn_next = Button(ax_next, '▶')

        def _clamp_scenario(val):
            # keep within [1, len(self.all_samples_data)]
            return max(1, min(len(self.all_samples_data), int(val)))

        def _scenario_step(delta):
            cur = int(scenario_slider.val)
            new_val = _clamp_scenario(cur + delta)
            if new_val != cur:
                # This will trigger on_scenario_change via the slider's callback
                scenario_slider.set_val(new_val)

        btn_prev.on_clicked(lambda event: _scenario_step(-1))
        btn_next.on_clicked(lambda event: _scenario_step(1))

        ax_vehicle = plt.axes([0.55, 0.04, 0.35, 0.03])
        self.vehicle_slider = Slider(ax=ax_vehicle, label='Active Vehicle', valmin=1, valmax=num_vehicles or 1,
                                     valinit=self.current_vehicle_idx + 1, valstep=1)

        ax_mode = plt.axes([0.80, 0.1, 0.15, 0.08])
        mode_radio = RadioButtons(ax_mode, MODES, active=MODES.index(self.sample_mode))
        # Re-run the model with current conditions & mask; do not touch goals or sliders
        def _resample_after_toggle():
            self.all_predictions.clear()
            # quiet and quick: no progress bar on toggles
            for _ in range(self.num_samples):
                self.all_predictions.extend(run_model())
            plot_all()

        # NEW: vehicle checkbox panel (left side)
        ax_imgmode = plt.axes([0.80, 0.20, 0.15, 0.08])  # sits above the mode radios
        img_modes = ["normal", "zero", "random"]
        img_radio = RadioButtons(ax_imgmode, img_modes, active=0)
        ax_imgmode.set_title("x_image", fontsize=10)

        def on_img_mode_change(label):
            self.image_mode = label  # "normal" | "zero" | "random"
            print(f"x_image mode set to: {label}")
            self.zero_image = label == "zero"
            if label == "random":
                self.random_image_idx = random.randint(0, len(self.all_samples_data))
                self.random_image = True
            else:
                self.random_image = False
            self.all_predictions.clear()
            for _ in range(self.num_samples):
                self.all_predictions.extend(run_model())
            plot_all()

        img_radio.on_clicked(on_img_mode_change)

        def _build_vehicle_panel():
            from matplotlib.widgets import CheckButtons
            if self.ax_panel is None:
                self.ax_panel = plt.axes([0.02, 0.22, 0.12, 0.70])  # left, bottom, width, height

            self.ax_panel.clear()
            sc = self.sample_cache
            num_v = self._get_num_vehicles_in_scenario(sc)
            labels = [str(i + 1) for i in range(num_v)]
            states = [bool(x) for x in self.include_mask]
            self.check_buttons = CheckButtons(self.ax_panel, labels, states)

            def _on_check(label):
                idx = int(label) - 1
                self.include_mask[idx] = not self.include_mask[idx]
                print(f"{'Included' if self.include_mask[idx] else 'Excluded'} vehicle {idx + 1}.")
                _resample_after_toggle()  # <- re-infer with same conditions

            self.check_buttons.on_clicked(_on_check)
            self.ax_panel.set_title("Vehicles", fontsize=10)
            for t in self.check_buttons.labels:
                t.set_fontsize(9)
            self.fig.canvas.draw_idle()

        _build_vehicle_panel()

        def on_temp_change(val):
            self.temperature = val
            print(f"Temp set to {val:.2f}. Click map to re-sample with new temperature.")

        def on_scenario_change(val):
            # switch scenario
            self.current_scenario_idx = int(val) - 1
            self.sample_cache = self.all_samples_data[self.current_scenario_idx]
            self.original_sample_cache = self.all_samples_data[self.current_scenario_idx]

            num_v = self._get_num_vehicles_in_scenario(self.sample_cache)

            # reset include mask for new scenario
            self.include_mask = np.ones(num_v, dtype=bool)

            # update vehicle slider range
            self.vehicle_slider.valmax = num_v or 1
            self.vehicle_slider.ax.set_xlim(1, num_v if num_v > 1 else 2)
            self.current_vehicle_idx = min(self.current_vehicle_idx, num_v - 1) if num_v > 0 else 0
            self.vehicle_slider.set_val(self.current_vehicle_idx + 1)

            print(f"Switched to scenario {self.current_scenario_idx + 1} ({num_v} vehicles).")

            # rebuild panel, clear predictions, redraw
            _build_vehicle_panel()
            self.all_predictions.clear()
            plot_all()

        def on_vehicle_change(val):
            self.current_vehicle_idx = int(val) - 1
            print(f"Switched active vehicle to {self.current_vehicle_idx + 1}. Click map to set its goal.")
            plot_all()

        def on_mode_change(label):
            self.sample_mode = label
            print(f"Sample mode set to {label}. Click map to re-sample.")

        temp_slider.on_changed(on_temp_change)
        scenario_slider.on_changed(on_scenario_change)
        self.vehicle_slider.on_changed(on_vehicle_change)
        mode_radio.on_clicked(on_mode_change)

        cid_click = self.fig.canvas.mpl_connect('button_press_event', on_click)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', on_key)

        print("\n--- Interactive Evaluator ---")
        print("- Click on the map to set a goal for the active vehicle and generate trajectories.")
        print("- Use the vehicle checkboxes (left panel) to include/exclude vehicles from the model inputs.")
        print("- Press 'h' to toggle whether excluded vehicles are hidden in the plot.")
        print("- Use the 'Active Vehicle' slider to choose which vehicle's goal to set.")
        print("- Press 'r' to clear all trajectories.")
        plt.show()

        self.fig.canvas.mpl_disconnect(cid_click)
        self.fig.canvas.mpl_disconnect(cid_key)
        print("Done.")
        return
