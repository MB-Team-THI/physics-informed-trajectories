import os
import logging
import pickle
import pprint
import random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')  # comment out if running headless
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 3rd party / project imports
from matplotlib.widgets import RadioButtons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.evaluation.eval import eval
from src.utils.average_meter import AverageMeter
from src.utils.get_fde import get_fde
from src.utils.get_ade import get_ade, get_displacement_per_timestep
from src.utils.rot_points import rot_points
from einops import rearrange, repeat
from src.utils.get_cmap import get_cmap
from src.utils.get_img_from_fig import get_img_from_fig
# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
NUM_SAMPLES = None
BEST_DIFF = 900
BEST_TAU = None

import os
def normalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return (t - val_min) / (val_max - val_min)

def denormalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return t * (val_max - val_min) + val_min

def _squeeze_traj_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Input shape expected: [B, N, 1, T] (as produced by _process_dynamic_model).
    Returns: [B, N, T]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,N,1,T]; got {x.shape}")
    return x.squeeze(2)


def _make_valid_mask(gt_x: torch.Tensor, gt_y: torch.Tensor) -> torch.Tensor:
    """
    gt_x, gt_y: [B,N,T]
    Returns: bool mask [B,N,T] where both X and Y are finite.
    """
    return ~(torch.isnan(gt_x) | torch.isnan(gt_y))
def _rep(data):
    if NUM_SAMPLES == 1:
        return data
    if isinstance(data, torch.Tensor):
        data = repeat(data, 'b ... -> (n b) ... ', n=NUM_SAMPLES)
        return data
    elif isinstance(data, list):
        return data * NUM_SAMPLES


# --------------------------------------------------------------------------------------
# Multi-sample metrics
# --------------------------------------------------------------------------------------

def _compute_ade_fde(
    pred_x_samples,
    pred_y_samples,
    gTruthX: torch.Tensor,
    gTruthY: torch.Tensor,
    x_traj_pred_len,
    maskXnot: torch.Tensor,
    maskYnot: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterate over each stochastic prediction sample and compute ADE & FDE
    by calling the project-provided `get_ade()` and `get_fde()` utilities.

    Args:
        pred_x_samples: List[Tensor] or Tensor
            Each sample prediction for X coordinates.
            Acceptable shapes per sample: [B,N,T] or [B,N,1,T].
            If a stacked Tensor is given: [S,B,N,T] or [S,B,N,1,T].
        pred_y_samples: List[Tensor] or Tensor
            Same as pred_x_samples, but for Y coordinates.
        gTruthX, gTruthY: torch.Tensor
            Ground-truth with shape [B,N,1,T] (as in your existing code).
        x_traj_pred_len: whatever you already pass into `get_fde` in your single-sample flow
            (often a list/array of per-batch sequence lengths).
        maskXnot, maskYnot: torch.Tensor
            Boolean masks in the same shape as gTruth tensors, as used in your
            single-sample metric calls.

    Returns:
        ade_per_sample: torch.Tensor [S]
        fde_per_sample: torch.Tensor [S]
            One scalar ADE / FDE per stochastic draw.
    """

        # Assume already a list
    px_list = pred_x_samples[0]
    py_list = pred_y_samples[0]
    S = px_list.shape[0]


    ade_vals = []
    fde_vals = []

    for s in range(S):
        px = px_list[s]
        py = py_list[s]

        px = px.unsqueeze(0)
        py = py.unsqueeze(0)

        # Compute metrics using your trusted utilities
        ade_s = get_ade(px, py, gTruthX, gTruthY, maskXnot, maskYnot, reduce=False)
        fde_s = get_fde(px, py, gTruthX, gTruthY, x_traj_pred_len, maskXnot, maskYnot, reduce=False)

        # `get_ade` / `get_fde` return scalars (per batch) in your current pipeline
        ade_vals.append(ade_s.unsqueeze(0))
        fde_vals.append(fde_s.unsqueeze(0))

    ade_per_sample = torch.cat(ade_vals, dim=0)  # [S]
    fde_per_sample = torch.cat(fde_vals, dim=0)  # [S]

    return ade_per_sample, fde_per_sample





import torch
from typing import Tuple, Union, List





# --------------------------------------------------------------------------------------
# Existing eval_730 (unchanged)
# --------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------
# Extended eval_753 for multi-modal evaluation
# --------------------------------------------------------------------------------------

class eval_753(eval):
    """
    Interactive scenario evaluation (original intent) + batch metrics calc.
    Extended to support multi-sample (multi-modal) metrics: minADE@5, minFDE@5,
    and average pairwise deviation across samples (diversity).
    """

    def __init__(self,
                 idx=121,
                 name='Interactive scenario evaluation',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=True,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 description='Interactive scenario: set new goal points for a chosen vehicle.',
                 save_dir: str = "./",   # metrics saved here
                 ):
        super().__init__(idx,
                         name,
                         input_,
                         output,
                         description)
        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model
        self.num_samples = NUM_SAMPLES
        self.save_dir = save_dir

        # internal references
        self.model = None
        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None
        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0
        self.target_len = 30  # default; can be overridden externally

        self.fig = None
        self.axs = None

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        self.model = model
        self.dataloader_test = dataloader_test
        # prefer caller's provided device; else auto
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dataset_dict = dataset_dict
        return self._evaluate(self.model, self.dataloader_test, self.device, self.dataset_dict)

    @staticmethod
    def _process_dynamic_model_reshape(output, gTruthX, gTruthY, gTruthT,
                                       x_traj_pred_obj_len, pres_object_lengths_sum):
        X = output['X']
        Y = output['Y']
        T = output['T']

        X_reshaped = torch.zeros_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.zeros_like(gTruthY, device=gTruthX.device)
        T_reshaped = torch.zeros_like(gTruthT, device=gTruthT.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = X[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp + 1], :
            ].unsqueeze(2)
            Y_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = Y[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp + 1], :
            ].unsqueeze(2)
            T_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = T[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp + 1], :
            ].unsqueeze(2)

        X_reshaped.copy_(gTruthX)
        Y_reshaped.copy_(gTruthY)
        T_reshaped.copy_(gTruthT)
        return X_reshaped, Y_reshaped, T_reshaped

    @staticmethod
    def _process_dynamic_model(output, gTruthX, gTruthY, gTruthT,
                               x_traj_pred_obj_len, pres_object_lengths_sum):
        X = output['X']
        Y = output['Y']
        T = output['T']

        X_reshaped = torch.empty_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.empty_like(gTruthY, device=gTruthY.device)
        T_reshaped = torch.empty_like(gTruthT, device=gTruthT.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = X[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp + 1], :
            ].unsqueeze(2)
            Y_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = Y[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp + 1], :
            ].unsqueeze(2)
            T_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = T[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp + 1], :
            ].unsqueeze(2)
        return X_reshaped, Y_reshaped, T_reshaped

    import torch

    @staticmethod
    def _compute_spread_metrics(pred_x_samples, pred_y_samples, valid_mask=None):
        # --- reshape to (S, N, seq) ---
        x = rearrange(pred_x_samples, '1 S N seq 1 -> S N seq')  # preserve named dims usage
        y = rearrange(pred_y_samples, '1 S N seq 1 -> S N seq')

        S, N, T = x.shape
        device = x.device
        dtype = x.dtype
        # valid mask setup
        if valid_mask is None:
            valid_mask = torch.ones((N, T), dtype=torch.bool, device=device)
        else:

            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            valid_mask = valid_mask.squeeze(-1)[0]
            if valid_mask.shape != (N, T):
                raise ValueError(f"valid_mask must have shape (N, seq) = ({N}, {T}), got {tuple(valid_mask.shape)}")


        # --- compute unbiased (sample) covariance entries across S ---
        # means over S
        mu_x = x.mean(dim=0)  # (N, T)
        mu_y = y.mean(dim=0)  # (N, T)

        # centered
        xc = x - mu_x  # (S, N, T)
        yc = y - mu_y  # (S, N, T)

        denom = max(1, S - 1)  # unbiased; guard S==1

        var_x_t = (xc * xc).sum(dim=0) / denom  # (N, T)
        var_y_t = (yc * yc).sum(dim=0) / denom  # (N, T)

        # --- 2D RMS spread per (N,T): sqrt(trace(Sigma_t)) ---
        std2d_t = (var_x_t + var_y_t).clamp_min(0).sqrt()  # (N, T)

        # --- determinant per (N,T) for ellipse areas ---


        # --- time-averaged (masked) per-vehicle metrics ---
        # mask as float for averaging
        m = valid_mask.to(dtype=dtype)  # (N, T)
        m_sum = m.sum(dim=1).clamp_min(1)  # (N,)

        per_vehicle_avg_std2d = (std2d_t * m).sum(dim=1) / m_sum  # (N,)


        # mean over vehicles where valid
        m_t = m  # (N, T)
        m_t_sum = m_t.sum(dim=0).clamp_min(1)  # (T,)
        per_timestep_avg_std2d = (std2d_t * m_t).sum(dim=0) / m_t_sum  # (T,)

        # --- overall scalars ---
        overall_mean_std2d = per_vehicle_avg_std2d.mean().item()


        metrics = {
            "per_vehicle_avg_std2d": per_vehicle_avg_std2d.mean().item(),  # (N,)
            "overall_mean_std2d": overall_mean_std2d,  # float

        }

        return metrics, per_timestep_avg_std2d

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

    @staticmethod
    def _compute_pairwise_diversity(pred_x_samples, pred_y_samples, valid_mask=None):
        """
        Computes the average pairwise diversity (L2 distance stddev) across S generations
        for each vehicle trajectory (N) and timestep (seq).

        Args:
            pred_x_samples: Tensor of shape (S, 1, N, seq, dim)
            pred_y_samples: Tensor of shape (S, 1, N, seq, dim)
            valid_mask:     Unused, included for compatibility

        Returns:
            diversity: scalar float (averaged over N and seq)
            per_timestep_diversity: Tensor of shape (seq,) with mean stddev at each timestep
        """
        # Merge and rearrange samples

        x = rearrange(pred_x_samples, '1 S  N seq 1 -> S N seq')
        y = rearrange(pred_y_samples, '1 S  N seq 1 -> S N seq')

        # Stack into a (S, N, seq, 2) tensor
        coords = torch.stack([x, y], dim=-1)  # shape: (S, N, seq, 2)

        # Compute stddev across samples: shape (N, seq, 2)
        std = coords.std(dim=0, unbiased=False)

        # Compute per-timestep diversity: mean over N (vehicles) and dim (x,y)
        per_timestep_diversity = std.mean(dim=(0, 2))  # shape: (seq,)

        # Compute overall diversity: mean over all dimensions
        diversity = std.mean().item()

        return diversity, per_timestep_diversity

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def float_range(min_val: float, max_val: float, step: float) -> list[float]:
        """Return a list [min_val, min_val+step, ..., max_val] inclusive."""
        n_steps = int(round((max_val - min_val) / step))
        return [round(min_val + i * step, 10) for i in range(n_steps + 1)]

    def _evaluate(self, model, dataloader_test, device, dataset_dict):
        """
        Sweep loop over sampling config combinations.
        Automatically runs _evaluate_fn() for all variants.
        """
        from itertools import product

        sample_counts = [64]
        batch_limits = [100]
        temperatures = [1.0]
        sampling_modes = ["multinomial"]

        results_all = []

        for ns, nb, temp, mode in product(sample_counts, batch_limits, temperatures, sampling_modes):
            # Override module-level globals
            global NUM_SAMPLES, NUM_BATCHES, TEMPERATURE, SAMPLING_MODE
            NUM_SAMPLES = ns
            NUM_BATCHES = nb
            TEMPERATURE = temp
            SAMPLING_MODE = mode

            # Use a unique save dir per config

            print(f"[EVAL] Running with NUM_SAMPLES={ns}, NUM_BATCHES={nb}, T={temp}, MODE={mode}")
            result = self._evaluate_fn(model, dataloader_test, device, dataset_dict)


    def _evaluate_fn(self, model, dataloader_test, device, dataset_dict, frame_idx=0):
        """
        Runs through the test loader, draws multiple samples per batch,
        computes ADE/FDE (single-sample), minADE@5/minFDE@5 (multi-sample),
        and average pairwise deviation across samples.
        """
        if model is None or dataloader_test is None:
            raise ValueError("Model and dataloader_test must be provided.")

        model.eval()
        if next(model.parameters()).device is not device:
            model.to(device)

        ade_list_single: List[float] = []
        fde_list_single: List[float] = []

        # Multi-sample accumulators (flattened over dataset)
        ade_min5_all: List[float] = []
        fde_min5_all: List[float] = []
        ade_max5_all: List[float] = []
        fde_max5_all: List[float] = []
        ade_avg2_all: List[float] = []
        fde_avg2_all: List[float] = []
        diversity_ade_all: List[float] = []
        diversity_fde_all: List[float] = []
        diversity_all: List[float] = []
        diversity_timestep_all: List[torch.Tensor] = []
        overall_mean_std2d: List[float] = []




        # Iterate test data
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            # Optional break to keep runtime manageable (match your prior code)
            if batch_idx > NUM_BATCHES:
                break

            # Assume batch size may be >1; original code asserted 1
            B = sample['pred_objsx'].shape[0]
            self.sample_cache = sample

            # Prepare inputs
            x_image = self.sample_cache['images'].to(self.device)

            x_traj = [
                self.sample_cache['hist_objs'].to(self.device),
                self.sample_cache['hist_obj_lens']
            ]
            x_traj_len = self.sample_cache['hist_objs_seq_len']
            x_traj_pred_obj_len = self.sample_cache['pred_obj_lens']
            x_traj_pred_len = self.sample_cache['pred_objs_seq_len']
            pres_object_lengths_sum = self.sample_cache['pres_object_lengths_sum']
            obj_length_padded = self.sample_cache['hist_object_lengths_sum']

            batch_wise_decoder_input = self.sample_cache['obj_decoder_in'].to(self.device)
            gTruthX = self.sample_cache['pred_objsx'].to(self.device)
            gTruthY = self.sample_cache['pred_objsy'].to(self.device)
            gTruthT = self.sample_cache['pred_objst'].to(self.device, non_blocking=True)
            maskXnotSingle = torch.isnan(gTruthX)
            maskYnotSingle = torch.isnan(gTruthX)
            gTruthXSingle = gTruthX.clone()
            gTruthYSingle = gTruthY.clone()
            gTruthX = _rep(gTruthX)
            gTruthY = _rep(gTruthY)
            gTruthT = _rep(gTruthT)

            # Additional GT signals (not directly used below but kept for completeness)
            gt_v = self.sample_cache['pred_objsv'].to(self.device)
            gt_a_lat = self.sample_cache['pred_objs_a_lat'].to(self.device)
            gt_a_lon = self.sample_cache['pred_objs_a_lon'].to(self.device)
            gt_psi = self.sample_cache['pred_objspsi'].to(self.device)

            condition_goal_point = self.sample_cache['cond_goal_point'].to(self.device, non_blocking=True).float()
            condition_v = self.sample_cache['cond_v'].to(self.device, non_blocking=True).float()
            combined_condition = torch.cat([condition_goal_point], dim=2)

            # ------------------------------------------------------------------
            # Draw multiple samples from the model
            # (If your model supports a 'num_samples' arg, replace the loop.)
            # ------------------------------------------------------------------
            sample_preds_X: List[torch.Tensor] = []
            sample_preds_Y: List[torch.Tensor] = []
            sample_preds_T: List[torch.Tensor] = []


            with torch.no_grad():
                x_image = _rep(x_image)
                x_traj = [_rep(x_traj[0]), _rep(x_traj[1])]
                x_traj_len = _rep(x_traj_len)

                batch_wise_object_lengths_sum = torch.tensor([i * obj_length_padded[1] for i in range(NUM_SAMPLES + 1)])

                conditions = _rep(combined_condition)
                batch_wise_decoder_input = _rep(batch_wise_decoder_input)
                output = model(
                    x_image=x_image,
                    x_traj=x_traj,
                    x_traj_len=x_traj_len,
                    batch_wise_object_lengths_sum=batch_wise_object_lengths_sum, # ascending cum list
                    conditions=conditions,
                    batch_wise_decoder_input=batch_wise_decoder_input,
                    target_length=self.target_len,
                    temperature=TEMPERATURE,
                    sample_mode=SAMPLING_MODE,
                    # If your model has a sampling interface (e.g., z noise), add it here.
                )

            X_reshaped_s, Y_reshaped_s, T_reshaped_s = self._process_dynamic_model(
                output, gTruthX, gTruthY, gTruthT, x_traj_pred_obj_len * NUM_SAMPLES, batch_wise_object_lengths_sum
            )

            sample_preds_X.append(_squeeze_traj_dims(X_reshaped_s))  # [B,N,T]
            sample_preds_Y.append(_squeeze_traj_dims(Y_reshaped_s))
            sample_preds_T.append(_squeeze_traj_dims(T_reshaped_s))

            # sample preds of shape B, O, seq, 1
            pred_x_samples = torch.stack(sample_preds_X, dim=0)
            pred_y_samples = torch.stack(sample_preds_Y, dim=0)
            # pred_t_samples = torch.stack(sample_preds_T, dim=0)  # not used

            # Ground truth squeezed -> [B,N,T]
            gt_x_s = _squeeze_traj_dims(gTruthX)
            gt_y_s = _squeeze_traj_dims(gTruthY)
            valid_mask = _make_valid_mask(gt_x_s, gt_y_s)  # [B,N,T]


            # ------------------------------------------------------------------
            # Multi-sample metrics
            # ------------------------------------------------------------------
            ade_s, fde_s = _compute_ade_fde(
                sample_preds_X,  # list of [B,N,T]
                sample_preds_Y,
                gTruthXSingle,
                gTruthYSingle,
                x_traj_pred_len* 1,
                maskXnotSingle,
                maskYnotSingle,
            )  # -> [ROLLOUTS, VEHICLES]

            k = NUM_SAMPLES
            # min over samples per vehicle → mean over vehicles
            min_ade_per_vehicle, winners_ade = ade_s.min(dim=0)  # [N], [N]
            min_fde_per_vehicle, winners_fde = fde_s.min(dim=0)  # [N], [N]

            max_ade_per_vehicle, _ = ade_s.max(dim=0)  # [N], [N]
            max_fde_per_vehicle, _ = fde_s.max(dim=0)  # [N], [N]

            minADE = min_ade_per_vehicle.mean().item()
            minFDE = min_fde_per_vehicle.mean().item()

            maxADE= max_ade_per_vehicle.mean().item()
            maxFDE= max_fde_per_vehicle.mean().item()

            avgADE = ade_s.mean().item()
            avgFDE = fde_s.mean().item()
            ade_avg2_all.append(avgADE)
            fde_avg2_all.append(avgFDE)

            ade_max5_all.append(maxADE)
            fde_max5_all.append(maxFDE)

            ade_min5_all.append(minADE)
            fde_min5_all.append(minFDE)

            # Diversity (average pairwise ADE / FDE)
            diversity, diversity_per_timestep = self._compute_pairwise_diversity(
                pred_x_samples, pred_y_samples
            )  # [B,N]

            spreads = self._compute_spread_metrics(
                pred_x_samples,
                pred_y_samples,
                valid_mask
            )
            overall_mean_std2d.append(spreads[0]['overall_mean_std2d'])
            diversity_all.extend([diversity])
            diversity_timestep_all.append(diversity_per_timestep)



        # ----------------------------------------------------------------------------------
        # Convert accumulators to arrays
        # ----------------------------------------------------------------------------------
        ade_min5_array = np.array(ade_min5_all, dtype=np.float32)
        fde_min5_array = np.array(fde_min5_all, dtype=np.float32)
        ade_max5_all_array = np.array(ade_max5_all, dtype=np.float32)
        fde_max5_all_array = np.array(fde_max5_all, dtype=np.float32)
        overall_mean_std2d_array = np.array(overall_mean_std2d, dtype=np.float32)
        ade_avg = np.array(ade_avg2_all, dtype=np.float32)
        fde_avg = np.array(fde_avg2_all, dtype=np.float32)

        diversity_all_timestep_array = torch.stack(diversity_timestep_all, dim=0).mean(0).cpu().numpy()

        # div_ade_array = np.array(diversity_ade_all, dtype=np.float32)
        # div_fde_array = np.array(diversity_fde_all, dtype=np.float32)

        # ----------------------------------------------------------------------------------
        # Save metrics
        # ----------------------------------------------------------------------------------
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, "minade5_array.npy"), ade_min5_array)
        np.save(os.path.join(self.save_dir, "minfde5_array.npy"), fde_min5_array)
        np.save(os.path.join(self.save_dir, "diversity.npy"), diversity_all)
        np.save(os.path.join(self.save_dir, "diversity_timestep.npy"), diversity_all_timestep_array)
        np.save(os.path.join(self.save_dir, "maxade5_array.npy"), ade_max5_all_array)
        np.save(os.path.join(self.save_dir, "maxfde5_array.npy"), fde_max5_all_array)
        np.save(os.path.join(self.save_dir, "overall_mean_std2d_array.npy"), overall_mean_std2d_array)

        # np.save(os.path.join(self.save_dir, "diversity_ade_array.npy"), div_ade_array)
        # np.save(os.path.join(self.save_dir, "diversity_fde_array.npy"), div_fde_array)

        # ----------------------------------------------------------------------------------
        # Means
        # ----------------------------------------------------------------------------------

        minade5_mean = np.nanmean(ade_min5_array)
        minfde5_mean = np.nanmean(fde_min5_array)
        maxade5_mean = np.nanmean(ade_max5_all_array)
        maxfde5_mean = np.nanmean(fde_max5_all_array)
        avgade = np.nanmean(ade_avg)
        avgfde = np.nanmean(fde_avg)
        diversity_mean = np.nanmean(diversity_all)
        overall_mean_std2d_mean = np.nanmean(overall_mean_std2d_array)


        # ----------------------------------------------------------------------------------
        # ADE per timestep (single-sample)
        # ----------------------------------------------------------------------------------
        # ade_timestep_list_single: list of [B,?,T] but from get_displacement_per_timestep we had shape [B,N,1,T]
        # ----------------------------------------------------------------------------------
        # Return metrics dict
        # ----------------------------------------------------------------------------------
        # set best diff

        results = {
            f"maxADE@{NUM_SAMPLES}_mean": float(maxade5_mean),
            f"maxFDE@{NUM_SAMPLES}_mean": float(maxfde5_mean),
            f"minADE@{NUM_SAMPLES}_mean": float(minade5_mean),
            f"minFDE@{NUM_SAMPLES}_mean": float(minfde5_mean),
            f"avgADE@{NUM_SAMPLES}_mean": float(avgade),
            f"avgFDE@{NUM_SAMPLES}_mean": float(avgfde),
            # "diversity": float(diversity_mean),
            "RMS": float(overall_mean_std2d_mean),
            # "diversity_per_timestep": diversity_all_timestep_array.tolist(),
            "num_samples": self.num_samples,
            "temperature": TEMPERATURE,
            # "save_dir": self.save_dir,
            # "sampling_mode": SAMPLING_MODE,
        }
        pprint.pprint(results)
        return results
