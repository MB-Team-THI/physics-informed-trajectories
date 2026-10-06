import torch
def get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY, x_traj_pred_len, maskXnot, maskYnot, onlyEgo=False,reduce=True):
      if onlyEgo:
            assert False
      pred_fde_incides = torch.cumsum(torch.Tensor(x_traj_pred_len),0).long()
      pred_fde_incides = pred_fde_incides.to(X_reshaped.device)
      x_fde_pred = X_reshaped[~maskXnot]
      x_fde_pred = torch.take(x_fde_pred, pred_fde_incides-1)
      y_fde_pred = Y_reshaped[~maskXnot]
      y_fde_pred = torch.take(y_fde_pred, pred_fde_incides-1)
      x_fde_true = gTruthX[~maskXnot]
      x_fde_true = torch.take(x_fde_true, pred_fde_incides-1)
      y_fde_true = gTruthY[~maskXnot]
      y_fde_true = torch.take(y_fde_true, pred_fde_incides-1)

      fde_per_vehicle = torch.sqrt((x_fde_pred - x_fde_true) ** 2 + (y_fde_pred - y_fde_true) ** 2)

      if reduce:
            return fde_per_vehicle.mean()
      else:
            return fde_per_vehicle
