import torch
from einops import rearrange
def get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY, maskXnot, maskYnot, reduce=True):
    X_reshaped[maskXnot] = torch.nan
    Y_reshaped[maskYnot] = torch.nan

    pred_traj = torch.cat((X_reshaped, Y_reshaped), dim=3)
    gTruth_traj = torch.cat((gTruthX, gTruthY), dim=3)

    # Flatten batch and object dims
    pred_traj_flat = rearrange(pred_traj, 'B O T F -> (B O) T F')
    gTruth_traj_flat = rearrange(gTruth_traj, 'B O T F -> (B O) T F')

    # ADE per trajectory (mean over time, not yet reduced)
    ade_per_traj = torch.nanmean(
        torch.sqrt(torch.sum((pred_traj_flat - gTruth_traj_flat) ** 2, dim=2)),
        dim=1
    )

    if reduce:
        return torch.nanmean(ade_per_traj)  # scalar
    else:
        return ade_per_traj                 # [num_trajectories]


def get_displacement_per_timestep(X_reshaped, Y_reshaped, gTruthX, gTruthY, maskXnot, maskYnot):
    # Apply masks: set invalid values to NaN
    X_reshaped[maskXnot] = torch.nan
    Y_reshaped[maskYnot] = torch.nan

    # Concatenate X and Y into a single trajectory tensor
    pred_traj = torch.cat((X_reshaped, Y_reshaped), dim=3)  # Shape: (B, O, T, 2)
    gTruth_traj = torch.cat((gTruthX, gTruthY), dim=3)      # Shape: (B, O, T, 2)

    # Compute displacement (Euclidean distance) per timestep
    displacement = torch.sqrt(torch.sum((pred_traj - gTruth_traj) ** 2, dim=3))  # Shape: (B, O, T)

    return displacement  # Returns per-timestep displacement error
