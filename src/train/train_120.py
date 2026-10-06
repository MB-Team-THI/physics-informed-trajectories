import itertools
import logging
import os
from collections import OrderedDict
from datetime import datetime

import torch.multiprocessing as mp
from torch import nn
from torch.special import gammaln

from src.model.model_730_utils import DynamicLossBalancer
from src.train.train import Training
from src.utils.average_meter import AverageMeter
from src.evaluation.eval_730 import eval_730
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
from collections import defaultdict, OrderedDict
from scipy.stats import spearmanr

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.get_fde import get_fde
from src.utils.get_ade import get_ade
from src.utils.get_img_from_fig import get_img_from_fig
from torch.cuda.amp import  GradScaler
from torch.amp import autocast

import wandb
import torch, sys, platform
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("Python:", sys.version)
print("Platform:", platform.platform())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


#################
# HYPERPARAMETERS
#################

WARMUP_STEPS = 0
START_TEMP = 1.0
END_TEMP = 1.0
TEMP_SCHEDULER_STEPS = 500
LR = 0.0001
OVERFIT_SINGLE_BATCH = False
ACC_STEPS = 1
LOG_INTERVAL = 1
SAVE_INTERVAL = 1
EPOCH_OFFSET = 0
EVAL_IMAGES_TO_GENERATE = 0
WANDB_MODEL_LOG = 50
if os.name == 'nt':
    print("This OS is Windows.")
    DISABLE_WANDB = True
else:
    print("This OS is not Windows.")
    DISABLE_WANDB = False

PROXIMITY_LOSS_MODE = "agnostic" # 'sync', 'window', or 'agnostic'
USE_MIXED_PRECISION = True

USE_LOSS_POS = True
USE_LOSS_OFFROAD= False
USE_LOSS_PROXIMITY = True
USE_SMOOTHNESS = False

POS_LOSS_SCALE = 1.0
OFFROAD_LOSS_SCALE = 1.0
PROXIMITY_LOSS_SCALE = 1.0
SMOOTHNESS_LOSS_SCALE = 1.0

MOVE_TRESH = 5.0
USE_STATIONARY_EXCLUSION = True

ZERO_COND_PROB = 0.0


balancer = DynamicLossBalancer(beta=0.98, clamp=(0.25, 4.0), warmup=100)


def evidential_nig_loss(y, mu, lam, alpha, beta,
                        coeff: float = 1e-5,   # smaller = less collapse pressure
                        rescale: float = 10.0  # normalize residuals (meters -> ~units)
                        ):
    """
    y, mu, lam, alpha, beta: broadcastable (e.g. [S,T] or flat [N]).
    """
    eps = 1e-8
    # clamp away from singularities and ridiculous magnitudes
    lam   = lam.clamp(1e-4, 1e3)
    alpha = alpha.clamp(1.0 + 1e-4, 1e3)
    beta  = beta.clamp(1e-4, 1e3)

    r = (y - mu) / rescale
    two_bl = 2.0 * beta * (1.0 + lam)

    nll = (
        0.5 * torch.log(torch.pi / lam)
        + alpha * torch.log1p((r * r) * lam / two_bl)
        + (gammaln(alpha) - gammaln(alpha + 0.5))
        - alpha * torch.log(beta)
    )
    reg = coeff * torch.abs(r) * (2.0 * lam + alpha)
    loss = (nll + reg).mean()

    # safety net to keep training alive if numerics go bad
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=y.device, dtype=y.dtype) + 10.0
    return loss
def report_lipschitz(model: nn.Module, iters: int = 50, device: str = None, all=False):
    """
    Estimate per-layer Lipschitz constants and print them sorted (descending).
    - Handles: nn.Linear (power iteration on weight matrix)
    - Known activation bounds: ReLU/ReLU6/ELU/CELU/Hardtanh/LeakyReLU (<=1), Tanh (<=1), Sigmoid (<=0.25)
    - Everything else -> 'N/A'
    Returns: OrderedDict {module_name: (lip_value_or_None, type_name)}
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dtype = torch.float32  # compute in fp32 for stability

    def _normalize(x: torch.Tensor, eps: float = 1e-12):
        n = x.norm()
        if not torch.isfinite(n) or n < eps:
            x = torch.randn_like(x)
            n = x.norm() + eps
        return x / n

    @torch.no_grad()
    def _lip_linear(layer: nn.Linear, iters: int = 50) -> float:
        # Power iteration for spectral norm ||W||_2, W shape [out, in]
        W = layer.weight.detach().to(device=device, dtype=dtype)
        # v in R^{in}, u in R^{out}
        v = _normalize(torch.randn(W.shape[1], device=device, dtype=dtype))
        u = _normalize(torch.randn(W.shape[0], device=device, dtype=dtype))
        for _ in range(iters):
            u = _normalize(W @ v)       # u = W v / ||W v||
            v = _normalize(W.t() @ u)   # v = W^T u / ||W^T u||
        sigma = (u @ (W @ v)).item()    # Rayleigh quotient ≈ top singular value
        return float(abs(sigma))

    def _known_activation_lip(m: nn.Module):
        # Safe, standard pointwise bounds
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.ELU, nn.CELU, nn.Hardtanh)):
            return 1.0
        if isinstance(m, nn.LeakyReLU):
            return float(max(1.0, float(getattr(m, "negative_slope", 0.0))))
        if isinstance(m, nn.Tanh):
            return 1.0
        if isinstance(m, nn.Sigmoid):
            return 0.25
        return None  # unknown / skip

    results = []
    for name, module in model.named_modules():
        if name == "":  # skip root
            continue
        lip = None
        kind = type(module).__name__
        try:
            if isinstance(module, nn.Linear):
                lip = _lip_linear(module, iters=iters)
            else:
                lip = 0 #_known_activation_lip(module)
        except Exception:
            lip = None
        results.append((name, lip, kind))

    # Sort: finite values first (desc), N/A at the end
    finite = [(n, float(v), k) for (n, v, k) in results
              if isinstance(v, (int, float)) and torch.isfinite(torch.tensor(v))]
    finite_formatted = [(n + "_" + k,  v) for (n, v, k) in finite]
    finite_formatted = {k: v for k, v in finite_formatted}
    # Grouping/aggregation knobs
    group_depth = 2  # e.g., "encoder.autoregressive_decoder"
    agg = "max"  # one of: "max", "mean", "sum", "product"
    drop_levels = set()  # e.g., {"layers"} to collapse "model.layers.0..."

    # Bucket per prefix
    buckets = defaultdict(list)
    for name, val, _kind in finite:
        parts = [p for p in name.split(".") if p not in drop_levels]
        key = ".".join(parts[:group_depth]) if parts else name
        buckets[key].append(val)

    # Inline reducer
    def _reduce(vals):
        if agg == "max":
            return max(vals)
        elif agg == "mean":
            return sum(vals) / max(1, len(vals))
        elif agg == "sum":
            return sum(vals)
        elif agg == "product":
            p = 1.0
            for x in vals:
                p *= x
            return p
        return max(vals)  # default

    # Aggregate and sort (desc)
    grouped = sorted(((k, _reduce(vs)) for k, vs in buckets.items()),
                     key=lambda kv: kv[1], reverse=True)

    ordered = OrderedDict((k, v) for k, v in grouped)
    if not all:
        finite_formatted = {}
    ret = {**ordered, **finite_formatted}
    return {"Lipschitz/"+k :v for k,v in ret.items()}




def normalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return (t - val_min) / (val_max - val_min)

def get_linear_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0  # After warmup, keep base LR constant

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def linear_temperature_schedule(start_temp, end_temp, total_steps):
    def schedule(step):
        ratio = min(step / total_steps, 1.0)
        return start_temp + ratio * (end_temp - start_temp)
    return schedule

def log_grad_norms(modules: dict, writer, batch_idx: int, prefix: str = "GradNorm"):
    """
    Logs the L2 norm of gradients for given named modules to TensorBoard.

    Args:
        modules (dict): Dict of name -> nn.Module to log.
        writer (SummaryWriter): TensorBoard writer.
        batch_idx (int): Current global step or batch index.
        prefix (str): TensorBoard log prefix.
    """
    for name, module in modules.items():
        total_norm_sq = sum(
            (p.grad.norm(2).item() ** 2 for p in module.parameters() if p.grad is not None)
        )
        total_norm = total_norm_sq ** 0.5
        writer.add_scalar(f"{prefix}/{name}", total_norm, batch_idx)


class train_120(Training):

    def __init__(self, idx, **kwargs) -> None:
        super().__init__()
        self.description = "Driver Behaviour Model training logic"
        self.dataset_dict = kwargs['dataset_dict']
        self.summary_name = kwargs['summary_name']
        self.dynamic_model = kwargs['dynamic_model']
        self.frequency = kwargs['frequency']

    def run_training(self, model, dataloader_train, dataloader_test, loss_fc,
                     optimizer, scheduler, save_checkpoint):
        if dataloader_train.num_gpus > 1:
            mp.spawn(self._train_dist,
                     nprocs=dataloader_train.num_gpus,
                     args=(model, dataloader_train, dataloader_test, loss_fc,
                           optimizer, self.dataset_dict, save_checkpoint,
                           self.summary_name, self.dynamic_model))
        else:
            self._train(model, dataloader_train, dataloader_test, loss_fc,
                        optimizer, self.dataset_dict, save_checkpoint,
                        self.summary_name, self.dynamic_model)

    @staticmethod
    def _process_dynamic_model(output, gTruthX, gTruthY, gTruthT,
                               x_traj_pred_obj_len, pres_object_lengths_sum):
        X = output['X']
        Y = output['Y']
        T = output['T']

        X_reshaped = torch.empty_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.empty_like(gTruthY, device=gTruthX.device)
        T_reshaped = torch.empty_like(gTruthT, device=gTruthX.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = X[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[
                    unp + 1], :].unsqueeze(2)
            Y_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = Y[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[
                    unp + 1], :].unsqueeze(2)
            T_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = T[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[
                    unp + 1], :].unsqueeze(2)
        return X_reshaped, Y_reshaped, T_reshaped



    @staticmethod
    def _train(model, dataloader_train, dataloader_test, loss_fc, optimizer,
               dataset_dict, save_checkpoint, summary_name,
               dynamic_model) -> None:
        """Single gpu training
        """
        epochs = dataloader_train.epochs
        wandbmode = "online" if not DISABLE_WANDB else "disabled"
        wandb.init(project="driver-behavior", name=summary_name,mode=wandbmode,
                   config=dict(LR=LR, ACC_STEPS=ACC_STEPS, WARMUP_STEPS=WARMUP_STEPS, START_TEMP=START_TEMP, END_TEMP=END_TEMP, TEMP_SCHEDULER_STEPS=TEMP_SCHEDULER_STEPS, PROXIMITY_LOSS_SCALE=PROXIMITY_LOSS_SCALE))
        # optional: gradients/params tracking
        wandb.watch(model, log="all", log_freq=WANDB_MODEL_LOG)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        time_start = datetime.now()
        pbar = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                              dataloader_train.batch_size),
                    desc="init training...".center(50))
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=LR)
        warmup_steps = WARMUP_STEPS
        scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps)
        start_temp = START_TEMP
        end_temp = END_TEMP
        steps_per_epoch = len(dataloader_train.dataset) // dataloader_train.batch_size
        total_steps = epochs * steps_per_epoch
        temp_scheduler = linear_temperature_schedule(start_temp, end_temp, TEMP_SCHEDULER_STEPS)
        if type(loss_fc) is list:
            if len(loss_fc) == 2:
                mse_loss = loss_fc[0]
                offroad_loss = loss_fc[1]
            if len(loss_fc) == 3:
                mse_loss = loss_fc[0]
                offroad_loss = loss_fc[1]
                proximity_loss = loss_fc[2]
        resolution = np.array(dataset_dict['bbox_pixel']) / np.array(
            dataset_dict['bbox_meter'])
        seq_len = dataset_dict['hist_seq_last']
        center = dataset_dict['center_meter']
        min_eval_loss = 1000000000
        accumulation_steps = ACC_STEPS
        scaler = GradScaler()  # handles loss scaling
        print(F"Accumulation steps: {accumulation_steps}")


        for epoch in range(epochs):
            epoch += EPOCH_OFFSET
            dataloader = dataloader_train(epoch)
            model.train()
            visualize = True

            loss_record = AverageMeter()
            ade_record = AverageMeter()
            fde_record = AverageMeter()
            T_mse_record = AverageMeter()
            max_dy_1 = AverageMeter()
            max_dy_2 = AverageMeter()
            batch_pass = 0
            torch.cuda.empty_cache()

            overfit_single_batch = OVERFIT_SINGLE_BATCH  # Toggle this
            if overfit_single_batch:
                print("Overfitting on a single batch for debugging purposes.")

            if overfit_single_batch:
                single_batch = next(iter(dataloader))
                dataloop = itertools.repeat(single_batch)  # Infinite repetition of one batch
            else:
                dataloop = dataloader
            for batch_idx, sample in enumerate(dataloop, start=epoch * len(dataloader)):
                optimizer.zero_grad()
                # if batch_idx > 500:
                #     assert False

                x_image = sample['images'].to(device, non_blocking=True)
                x_traj = [
                    sample['hist_objs'].to(device, non_blocking=True),
                    sample['hist_obj_lens']
                ]
                y_traj_merge = torch.cat([sample["pred_objsx"], sample["pred_objsy"], sample["pred_objst"], sample["pred_objspsi"], sample["pred_objsv"]], dim=-1)
                y_traj = [
                    y_traj_merge.to(device, non_blocking=True),
                    sample['pred_obj_lens'],
                    sample['pres_object_lengths_sum']
                ]
                condition_goal_point = sample['cond_goal_point'].to(device, non_blocking=True).float().to(device, non_blocking=True)
                condition_v = sample['cond_v'].to(device, non_blocking=True).float().to(device, non_blocking=True)
                # condition_profile = sample['vehicle_class'].to(device, non_blocking=True).float().unsqueeze(-1)

                x_traj_len = sample['hist_objs_seq_len']
                x_traj_pred_obj_len = sample['pred_obj_lens']
                x_traj_pred_len = sample['pred_objs_seq_len']


                obj_length_padded = sample['hist_object_lengths_sum']
                pres_object_lengths_sum = sample['pres_object_lengths_sum']
                batch_wise_decoder_input = sample['obj_decoder_in'].to(
                    device, non_blocking=True)
                gTruthX = sample['pred_objsx'].to(device, non_blocking=True)
                gTruthY = sample['pred_objsy'].to(device, non_blocking=True)
                gTruthT = sample['pred_objst'].to(device, non_blocking=True)
                if dataset_dict['name'] == 'lyft':
                    target_len = 31
                else:
                    target_len = 30  # trajectory length as a parameter should be included in the training
                """
                x_image: 16,1,240,240
                x_traj: List[16,22,20,5; List[int]:16]
                x_traj_len: List[int]:183
                obj_length_padded: List[int]:17
                batch_wise_decoder_input: 16,22,5
                target_len: 30
                """
                # with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True,profile_memory=True) as prof:
                #     with record_function("model_inference"):
                combined_condition = torch.cat([condition_goal_point], dim=2)
                # B, T, D = combined_condition.shape
                # mask = (torch.rand(B, T, 1, device=combined_condition.device) > ZERO_COND_PROB).float()
                # combined_condition = combined_condition * mask

                temperature = temp_scheduler(batch_idx)
                with autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                    output = model(
                        x_image=x_image,
                        x_traj=x_traj,
                        y_traj= y_traj,
                        x_traj_len=x_traj_len,
                        batch_wise_object_lengths_sum=obj_length_padded,
                        batch_wise_decoder_input=batch_wise_decoder_input,
                        target_length=target_len,
                        temperature=temperature,
                        conditions=combined_condition)

                # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                # quit()
                    X_reshaped, Y_reshaped, T_reshaped = train_120._process_dynamic_model(
                        output, gTruthX, gTruthY, gTruthT, x_traj_pred_obj_len,
                        pres_object_lengths_sum)
                    maskXnot = torch.isnan(gTruthX)
                    maskYnot = torch.isnan(gTruthX)

                    B, N, T, D = X_reshaped.shape
                    assert D == 1 and T >= 1

                    # valid when BOTH x and y are valid
                    valid = (~maskXnot[..., 0]) & (~maskYnot[..., 0])  # [B,N,T]
                    t = torch.arange(T, device=X_reshaped.device)  # [T]

                    # first valid idx: invalid -> +T, take min
                    first_idx = torch.where(valid, t, t.new_full((T,), T)).min(dim=2).values  # [B,N]
                    # last valid idx: invalid -> -1, take max
                    last_idx = torch.where(valid, t, t.new_full((T,), -1)).max(dim=2).values  # [B,N]

                    # can check only if we have at least two valid points
                    can_check = (first_idx >= 0) & (first_idx < T) & (last_idx >= 0) & (last_idx > first_idx)

                    # gather endpoints
                    fi = first_idx.clamp(0, T - 1).unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]
                    li = last_idx.clamp(0, T - 1).unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]

                    x_first = torch.gather(X_reshaped, 2, fi).squeeze(-1).squeeze(-1)  # [B,N]
                    y_first = torch.gather(Y_reshaped, 2, fi).squeeze(-1).squeeze(-1)
                    x_last = torch.gather(X_reshaped, 2, li).squeeze(-1).squeeze(-1)
                    y_last = torch.gather(Y_reshaped, 2, li).squeeze(-1).squeeze(-1)

                    dist_tot = torch.sqrt((x_last - x_first) ** 2 + (y_last - y_first) ** 2)  # [B,N]

                    # include agent if total displacement > threshold; if we can't check, include it
                    include_agent = torch.where(can_check, dist_tot > MOVE_TRESH,
                                                torch.ones_like(can_check, dtype=torch.bool))  # [B,N]

                    agent_mask = include_agent[:, :, None, None]  # [B,N,1,1]

                    valid_X = (~maskXnot) & agent_mask if USE_STATIONARY_EXCLUSION else (~maskXnot)
                    valid_Y = (~maskYnot) & agent_mask if USE_STATIONARY_EXCLUSION else (~maskYnot)

                    X_mse = mse_loss(X_reshaped[valid_X], gTruthX[valid_X]) if valid_X.any() else X_reshaped.new_tensor(
                        0.0)
                    Y_mse = mse_loss(Y_reshaped[valid_Y], gTruthY[valid_Y]) if valid_Y.any() else Y_reshaped.new_tensor(
                        0.0)


                    # mu_X = output["mu_X_seq"]
                    # mu_Y = output["mu_Y_seq"]
                    # lam_X, alpha_X, beta_X = output["lam_X_seq"], output["alpha_X_seq"], output["beta_X_seq"]
                    # lam_Y, alpha_Y, beta_Y = output["lam_Y_seq"], output["alpha_Y_seq"], output["beta_Y_seq"]

                    gtX_slices, gtY_slices, mask_slices = [], [], []

                    for b in range(B):
                        n_obj = int(x_traj_pred_obj_len[b])  # num valid objects
                        gtX_b = gTruthX[b, :n_obj, :, 0]  # [n_obj, T]
                        gtY_b = gTruthY[b, :n_obj, :, 0]
                        mask_valid_b = (~torch.isnan(gtX_b)) & (~torch.isnan(gtY_b))  # [n_obj, T]

                        gtX_slices.append(gtX_b)
                        gtY_slices.append(gtY_b)
                        mask_slices.append(mask_valid_b)

                    # flatten across batch
                    gTruthX_flat = torch.cat(gtX_slices, dim=0)  # [S, T]
                    gTruthY_flat = torch.cat(gtY_slices, dim=0)  # [S, T]
                    valid_mask = torch.cat(mask_slices, dim=0)  # [S, T]

                    yX = gTruthX_flat[valid_mask]
                    yY = gTruthY_flat[valid_mask]
                    # muX = mu_X[valid_mask]
                    # muY = mu_Y[valid_mask]
                    # lamX = lam_X[valid_mask]
                    # lamY = lam_Y[valid_mask]
                    # alphaX = alpha_X[valid_mask]
                    # alphaY = alpha_Y[valid_mask]
                    # betaX = beta_X[valid_mask]
                    # betaY = beta_Y[valid_mask]

                    # # Now compute evidential losses
                    # nig_loss_X = evidential_nig_loss(yX, muX, lamX, alphaX, betaX, coeff=1e-6)
                    # nig_loss_Y = evidential_nig_loss(yY, muY, lamY, alphaY, betaY, coeff=1e-6)

                    # lambda_evi = 0.3
                    # evidence_loss = lambda_evi * (nig_loss_X + nig_loss_Y)


                    # b = max(0, min(0, X_reshaped.shape[0] - 1))
                    # Xb = X_reshaped[b, :, :, 0].detach().cpu().numpy()  # [N,T]
                    # Yb = Y_reshaped[b, :, :, 0].detach().cpu().numpy()  # [N,T]
                    # valid_b = ((~maskXnot[b, :, :, 0]) & (~maskYnot[b, :, :, 0])).detach().cpu().numpy()
                    # inc = include_agent[b].detach().cpu().numpy()  # [N]
                    #
                    # # colors
                    # c_inc = "tab:blue"  # included (used in loss)
                    # c_exc = "tab:red"  # excluded (stationary)
                    #
                    # plt.figure(figsize=(6, 6))
                    # for n in range(Xb.shape[0]):
                    #     pts_ok = valid_b[n] & np.isfinite(Xb[n]) & np.isfinite(Yb[n])
                    #     if pts_ok.sum() < 2:
                    #         continue
                    #     color = c_inc if inc[n] else c_exc
                    #     plt.plot(Xb[n, pts_ok], Yb[n, pts_ok], color=color, linewidth=1.8, alpha=0.9)
                    #
                    # plt.gca().set_aspect("equal", adjustable="box")
                    # plt.xlabel("X [m]")
                    # plt.ylabel("Y [m]")
                    # plt.title(f"Trajectories (batch {b}) — blue=included, red=excluded")
                    # plt.grid(True, linestyle=":", linewidth=0.5)
                    # plt.tight_layout()
                    # plt.show()


                    proximity_loss_val = proximity_loss(output["X"], output["Y"], x_traj_pred_obj_len,
                                                        valid_mask) * PROXIMITY_LOSS_SCALE

                    offroad_loss_val = offroad_loss(output["X"], output["Y"], x_traj_pred_obj_len, valid_mask,x_image,center, resolution)
                    smoothness_targets = [output["psi_dot"], output["ax"]]
                    scaling_factors = [0.1,0.1]  # Each lambda corresponds to the target at the same index

                    total_smoothness_loss = 0.0

                    for target, lambda_smooth in zip(smoothness_targets, scaling_factors):
                        # Compute differences between consecutive elements along the time axis (axis 1)
                        differences = target[:, 1:] - target[:, :-1]

                        # Compute the smoothness penalty as the squared L2 norm (mean squared differences)
                        smoothness_penalty = torch.mean(differences ** 2)

                        # Scale the penalty by its corresponding lambda and add it to the total loss
                        total_smoothness_loss += lambda_smooth * smoothness_penalty


                    cond_v = condition_v # 128, 23, 1
                    out_v = output["v"] # 1233, 30
                    cond_v_flat = cond_v.view(-1)  # Shape: [128 * 23 * 1] = [2944]
                    # Combine masks and remove the last singleton dimension
                    combined_mask = (~maskXnot & ~maskYnot).squeeze(-1)  # Shape: [128, 23, 30]

                    # Reduce over the time dimension to identify valid positions in `cond_v`
                    valid_cond_mask = combined_mask.any(dim=2)  # Shape: [128, 23]

                    # Flatten the mask to match the shape of `cond_v_flat`
                    valid_cond_mask_flat = valid_cond_mask.view(-1)  # Shape: [128 * 23] = [2944]
                    valid_cond_v = cond_v_flat[valid_cond_mask_flat]  # Shape: [1233]
                    avg_out_v = out_v.mean(dim=1)  # Shape: [1233]


                    loss_v = mse_loss(avg_out_v, valid_cond_v)

                    if dataset_dict['name'] == 'lyft':
                        gTruthT = gTruthT * 1e-9
                    T_mse = mse_loss(T_reshaped[~maskXnot], gTruthT[~maskYnot])
                    # delta_loss_val, max_delta_1, max_delta_2 = delta_loss(
                    #     output, dynamic_model)
                    term_pos = X_mse + Y_mse
                    term_pos = term_pos * POS_LOSS_SCALE


                    CFG = [
                        (term_pos, POS_LOSS_SCALE, USE_LOSS_POS),
                        (proximity_loss_val, PROXIMITY_LOSS_SCALE, USE_LOSS_PROXIMITY),
                        (offroad_loss_val, OFFROAD_LOSS_SCALE, USE_LOSS_OFFROAD),
                        (total_smoothness_loss, SMOOTHNESS_LOSS_SCALE, USE_SMOOTHNESS),
                    ]
                    NAMES = ["pos", "prox", "offroad", "smoothness"]

                    _, dbg = balancer(CFG, names=NAMES)
                    loss = term_pos + total_smoothness_loss




                    scaler.scale(loss / accumulation_steps).backward()
                    # scale = 1 / 30
                    # for p in model.encoder.autoregressive_decoder.parameters():
                    #     if p.grad is not None:
                    #         p.grad.mul_(scale)
                    # # report gradient existence for each encoder
                    # for name, enc in [("encoder_image", model.encoder.encoder_image),
                    #                   ("merger", model.encoder.merger_transformer),
                    #                   ("encoder_trajectory", model.encoder.encoder_trajectory)]:
                    #     grads = [p.grad for p in enc.parameters() if p.grad is not None]
                    #     has_grad = any(g.abs().sum() > 0 for g in grads)
                    #
                    #     print(f"{name}: {'✅ gradient' if has_grad else '❌ no gradient'}")
                    total_norm = 0.0

                    if (batch_idx + 1) % accumulation_steps == 0:
                        # 1) unscale grads so clipping works in real units
                        scaler.unscale_(optimizer)

                        # 2) clip global L2 norm
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

                        # (optional) guard against NaNs/Infs
                        if not torch.isfinite(total_norm):
                            optimizer.zero_grad(set_to_none=True)
                            print(f"Warning: non-finite gradient norm {total_norm}. Skipping step.")
                            scaler.update()  # still advance internal scale
                            continue

                        # 3) optimizer step via scaler
                        scaler.step(optimizer)
                        scaler.update()

                        # 4) LR schedule and zero grads
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    ade = get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY,
                                  maskXnot, maskYnot)
                    fde = get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY,
                                  x_traj_pred_len, maskXnot, maskXnot)


                psi_dot = output['psi_dot']

                # 1) First difference (like "acceleration" in heading)
                acc_psi = psi_dot[:, 1:] - psi_dot[:, :-1]  # shape (n, T-1)

                # 2) Second difference (like "jerk" in heading)
                jerk_psi = acc_psi[:, 1:] - acc_psi[:, :-1]  # shape (n, T-2)

                # 3) Mean absolute jerk for each trajectory
                mean_abs_jerk = torch.sum(torch.abs(jerk_psi), dim=1)  # shape (n,)



                # 4) Average across all trajectories
                avg_mean_abs_jerk = torch.mean(mean_abs_jerk)
                emas = dbg.get("emas")
                wandb.log({"Train_loss": loss.item(), "X_mse": X_mse.item(), "Y_mse": Y_mse.item(),
                           "Proximity Loss" : proximity_loss_val.item(),
                           "Offroad Loss": offroad_loss_val.item(),
                           "Total Grad Norm": total_norm,
                           "Total Smoothness Loss": total_smoothness_loss.item(),
                           "LossBalancer/positional_ema": emas.get("pos", 0),
                            "LossBalancer/proximity_ema": emas.get("prox", 0),
                            "LossBalancer/offroad_ema": emas.get("offroad", 0),
                            "LossBalancer/smoothness_ema": emas.get("smoothness", 0),
                           "ADE": ade.item(), "FDE": fde.item(), "FDE/ADE Ratio": fde.item() / ade.item(),
                            "temperature": temperature,
                           "lr": optimizer.param_groups[0]["lr"]}, step=batch_idx)


                wandb.log({
                    "acceleration_distribution": wandb.Histogram(output['ax'].detach().cpu().numpy()),
                    "psi_dot_distribution": wandb.Histogram(output['psi_dot'].detach().cpu().numpy()),
                    "v_distribution": wandb.Histogram(output['v'].detach().cpu().numpy()),
                }, step=batch_idx)

                fde_record.update(fde.item(), x_image.size(0))
                T_mse_record.update(T_mse.item(), x_image.size(0))
                ade_record.update(ade.item(), x_image.size(0))
                loss_record.update(loss.item(), x_image.shape[0])

                del X_mse
                del Y_mse
                del loss
                del output


                with torch.no_grad():
                    if visualize and epoch > 0 and epoch % LOG_INTERVAL == 0 or (overfit_single_batch and batch_idx % 500 ==0 and batch_idx > 0):
                        if overfit_single_batch:
                            save_checkpoint(epoch)
                        visualize = False
                        # take first batch of test data
                        for dataloader_i, name in [(dataloader_test, 'Test'), (dataloader_train, 'Train')]:
                            sample = next(iter(dataloader_i(epoch)))
                            x_image = sample['images'].to(device, non_blocking=True)
                            x_traj = [
                                sample['hist_objs'].to(device, non_blocking=True),
                                sample['hist_obj_lens']
                            ]
                            condition = sample['cond_goal_point'].to(device, non_blocking=True).float().to(device, non_blocking=True)
                            x_traj_len = sample['hist_objs_seq_len']
                            x_traj_pred_obj_len = sample['pred_obj_lens']
                            x_traj_pred_len = sample['pred_objs_seq_len']
                            obj_length_padded = sample['hist_object_lengths_sum']
                            pres_object_lengths_sum = sample['pres_object_lengths_sum']
                            batch_wise_decoder_input = sample['obj_decoder_in'].to(
                                device, non_blocking=True)
                            gTruthX = sample['pred_objsx'].to(device, non_blocking=True)
                            gTruthY = sample['pred_objsy'].to(device, non_blocking=True)
                            gTruthT = sample['pred_objst'].to(device, non_blocking=True)
                            condition_goal_point = sample['cond_goal_point'].to(device, non_blocking=True).float().to(
                                device, non_blocking=True)
                            condition_v = sample['cond_v'].to(device, non_blocking=True).float().to(device,
                                                                                                    non_blocking=True)
                            # condition_profile = sample['vehicle_class'].to(device, non_blocking=True).float().unsqueeze(-1)
                            combined_condition = torch.cat([condition_goal_point], dim=2)

                            output = model(
                                x_image=x_image,
                                x_traj=x_traj,
                                x_traj_len=x_traj_len,
                                batch_wise_object_lengths_sum=obj_length_padded,
                                batch_wise_decoder_input=batch_wise_decoder_input,
                                target_length=target_len,
                                conditions=combined_condition)
                            X_reshaped, Y_reshaped, T_reshaped = train_120._process_dynamic_model(
                                output, gTruthX, gTruthY, gTruthT, x_traj_pred_obj_len,
                                pres_object_lengths_sum)
                            eval_visualizer = eval_730()
                            for i in range(EVAL_IMAGES_TO_GENERATE):
                                fig = eval_visualizer._evaluate(model, dataloader_i, device, [dataset_dict,None], frame_idx=i)
                                assert fig is not None
                                # Convert the matplotlib figure to an image.
                                image_to_display = get_img_from_fig(fig)

                                # Log the image to TensorBoard.
                                wandb.log({f"[{name}]_Full_Plot_{i}": wandb.Image(image_to_display)}, step=batch_idx)

                                # Close the figure to free resources.
                                plt.close(fig)


                        model.train()
                        optimizer.zero_grad()

                batch_pass = batch_pass + 1
                pbar.update(1)
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:.3f} ADE: {:.3f} FDE: {:.3f} Temp: {:.5f}  lr: {:.5f}".format(
                    epoch + 1, epochs, batch_pass, len(dataloader),
                    round(loss_record.avg, 3), ade_record.avg, fde_record.avg,
                    temperature,
                    optimizer.param_groups[0]["lr"]).center(50)
                pbar.set_description(log_msg)

                logging.info(log_msg)
                wandb.log({
                    "Train BatchWise Loss": loss_record.avg,
                    "FDE": fde.item(),
                    "ADE": ade.item(),
                    "FDE/ADE Ratio": fde.item() / ade.item(),
                    "Max_dy_1": max_dy_1.avg,
                    "Max_dy_2": max_dy_2.avg,
                }, step=batch_idx)
                logging.info(log_msg)

            print('\nEpoch: {}/{} Train Loss: {:.3f}'.format(
                epoch + 1, epochs, loss_record.avg))
            if epoch % SAVE_INTERVAL == 0 and epoch > 0:
                #eval_loss,ade_eval,fde_eval = eval_func(model,dataloader_test,device,dataset_dict)
                # print('\nEpoch: {}/{} Test Loss: {:.3f}'.format(epoch + 1, epochs, eval_loss))

                save_checkpoint(epoch)
                # writer.add_scalar("Test Loss", eval_loss, epoch)
                # writer.add_scalar("Test ADE", ade_eval, epoch)
                # writer.add_scalar("Test FDE", fde_eval, epoch)

        pbar.set_description("Training finished {} (Total time: {})".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            datetime.now() - time_start).center(50))

        pbar.close()
        logging.info(
            "End of Training -- total time: {}".format(datetime.now() -
                                                       time_start))