import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.widgets import RadioButtons
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.evaluation.eval import eval
from src.utils.average_meter import AverageMeter
from src.utils.get_fde import get_fde
from src.utils.get_ade import get_ade
import cv2
from src.utils.rot_points import rot_points
from einops import rearrange
from src.utils.get_cmap import get_cmap
import logging
import matplotlib.pyplot as plt
from src.utils.get_img_from_fig import get_img_from_fig
import os
import seaborn as sns
import scipy.stats as stats
import imageio
from tqdm import tqdm
import matplotlib

from matplotlib import pyplot as plt
plt.switch_backend("TkAgg")

# from argoverse.evaluation.competition_util import generate_forecasting_h5
# from argoverse.evaluation import eval_forecasting


def normalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return (t - val_min) / (val_max - val_min)


class eval_730(eval):
    def __init__(self,
                 idx=120,
                 name='FDE ADE ITSC 25 paper plot',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=False,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 description='Calculates MSE,ADE,FDE for all vehicles in the batch',
                 ):
        super().__init__(idx,
                         name,
                         input_,
                         output,
                         description)
        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        return self._evaluate(model, dataloader_test, device, dataset_dict)

    @staticmethod
    def _process_dynamic_model_reshape(output, gTruthX, gTruthY, gTruthT,
                                       x_traj_pred_obj_len, pres_object_lengths_sum):
        X = output['X']
        Y = output['Y']
        T = output['T']

        # Initialize reshaped tensors with the same values as gTruth tensors
        X_reshaped = torch.zeros_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.zeros_like(gTruthY, device=gTruthX.device)
        T_reshaped = torch.zeros_like(gTruthT, device=gTruthT.device)

        # Perform reshaping operations without modifying values
        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = X[
                                                                pres_object_lengths_sum[unp]:pres_object_lengths_sum[
                                                                    unp + 1], :
                                                                ].unsqueeze(2)
            Y_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = Y[
                                                                pres_object_lengths_sum[unp]:pres_object_lengths_sum[
                                                                    unp + 1], :
                                                                ].unsqueeze(2)
            T_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = T[
                                                                pres_object_lengths_sum[unp]:pres_object_lengths_sum[
                                                                    unp + 1], :
                                                                ].unsqueeze(2)

        # Copy the original values into the reshaped tensors
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
        Y_reshaped = torch.empty_like(gTruthY, device=gTruthX.device)
        T_reshaped = torch.empty_like(gTruthT, device=gTruthT.device)

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

    @torch.no_grad()
    def _evaluate(self, model, dataloader_test, device, dataset_dict, frame_idx=0):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """

        def mse_eval(out_1=None, out_2=None):
            return F.mse_loss(out_1, out_2, reduction='mean')

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_record = AverageMeter()
        ade_record = AverageMeter()
        fde_record = AverageMeter()
        model.eval()
        if next(model.parameters()).device is not device:
            model.to(device)

        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            # For demonstration, let's just evaluate the first batch and break.
            if batch_idx > 0:
                break

            x_image = sample['images'].to(device)
            x_traj = [sample['hist_objs'].to(device), sample['hist_obj_lens']]
            x_traj_len = sample['hist_objs_seq_len']
            x_traj_pred_obj_len = sample['pred_obj_lens']
            x_traj_pred_len = sample['pred_objs_seq_len']
            pres_object_lengths_sum = sample['pres_object_lengths_sum']
            obj_length_padded = sample['hist_object_lengths_sum']
            batch_wise_decoder_input = sample['obj_decoder_in'].to(device)
            gTruthX = sample['pred_objsx'].to(device)
            gTruthY = sample['pred_objsy'].to(device)
            gTruthT = sample['pred_objst'].to(device, non_blocking=True)

            # Example: how to modify specific scenario's goal points
            target_len = 30
            frame = frame_idx

            # Run model inference
            condition_goal_point = sample['cond_goal_point'].to(device, non_blocking=True).float()
            condition_v = sample['cond_v'].to(device, non_blocking=True).float()
            combined_condition = torch.cat([condition_goal_point, condition_v], dim=2)

            output = model(
                x_image=x_image,
                x_traj=x_traj,
                x_traj_len=x_traj_len,
                batch_wise_object_lengths_sum=obj_length_padded,
                conditions=combined_condition,
                batch_wise_decoder_input=batch_wise_decoder_input,
                target_length=target_len
            )

            # Reshape for easy comparison with ground truth
            X_reshaped, Y_reshaped, T_reshaped = eval_730._process_dynamic_model(
                output, gTruthX, gTruthY, gTruthT, x_traj_pred_obj_len, pres_object_lengths_sum
            )

            # Example plotting for one scenario in the batch:
            for idx, image in enumerate(x_image):
                if idx != frame:
                    continue

                image = image[0, :, :].cpu().numpy()
                traj_hist = x_traj[0][idx].cpu().numpy()
                gTruthX_plot = gTruthX[idx].cpu().numpy()
                gTruthY_plot = gTruthY[idx].cpu().numpy()
                X_reshaped_plot = X_reshaped[idx].cpu().numpy()
                Y_reshaped_plot = Y_reshaped[idx].cpu().numpy()

                # (Your normal plotting code here...)
                # Return or show a figure, or anything needed.

                fig, axs = plt.subplots(1, 1, figsize=(8, 6))
                axs.imshow(image, cmap='gray')
                axs.set_title("Non-interactive evaluation example (eval_730)")
                plt.show()
                return fig  # Just return the figure for demonstration.

        # If there's no batch, just return None
        return None


################################################################################################
# New class eval_740 with interactive scenario selection & vehicle goal modification
################################################################################################

class eval_761(eval):

    def __init__(self,
                 idx=121,
                 name='Prediction vs GT. Action Space ITSC 25 paper plot',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=True,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 description='Interactive scenario: set new goal points for a chosen vehicle.'):
        super().__init__(idx,
                         name,
                         input_,
                         output,
                         description)
        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model

        # We'll store some references after calling __call__ so we can
        # manage state for interactive plotting
        self.model = None
        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None
        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0
        self.target_len = 30

        # We will store figure and axis references for repeated updates
        self.fig = None
        self.axs = None

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        """
        Called externally, just like the previous class. We'll do an interactive loop inside.
        """
        # Store references so we can use them in the interactive plot
        self.model = model
        self.dataloader_test = dataloader_test
        self.device = "cuda"
        self.dataset_dict = dataset_dict

        return self._evaluate(self.model, self.dataloader_test, self.device, self.dataset_dict)

    @staticmethod
    def _process_dynamic_model_reshape(output, gTruthX, gTruthY, gTruthT,
                                       x_traj_pred_obj_len, pres_object_lengths_sum):
        # Same helper as in eval_730
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
        # Same helper as in eval_730
        X = output['X']
        Y = output['Y']
        T = output['T']

        X_reshaped = torch.empty_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.empty_like(gTruthY, device=gTruthX.device)
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


    @torch.no_grad()
    def _evaluate(self, model, dataloader_test, device, dataset_dict, frame_idx=0):
        """
        Evaluates accuracy on linear model trained on upstream ssl model
        """

        def mse_eval(out_1=None, out_2=None):
            return F.mse_loss(out_1, out_2, reduction='mean')

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_record = AverageMeter()
        ade_record = AverageMeter()
        fde_record = AverageMeter()
        save_results = False
        output_path = "/home/schenker3/Desktop/SCENARIO-NET/output_path/"
        model.eval()
        max_ax = 0
        max_ay = 0
        max_jx = 0
        max_jy = 0
        if next(model.parameters()).device is not device:
            model.to(device)
        output_all = {}
        g_all = {}

        def pretty_print_plot_data_shapes(plot_data):
            from collections.abc import Mapping
            import numpy as np

            def get_shape(val):
                if isinstance(val, (list, tuple)):
                    try:
                        arr = np.array(val)
                        return arr.shape
                    except:
                        return f'list(len={len(val)})'
                elif isinstance(val, np.ndarray):
                    return val.shape
                elif torch.is_tensor(val):
                    return tuple(val.size())
                else:
                    return type(val).__name__

            def print_recursive(d, indent=0):
                for k, v in d.items():
                    prefix = " " * indent
                    if isinstance(v, Mapping):
                        print(f"{prefix}{k}/")
                        print_recursive(v, indent + 4)
                    else:
                        shape = get_shape(v)
                        print(f"{prefix}{k}: shape = {shape}")

            print("Plot Data Structure:")
            print_recursive(plot_data)
        p_all = {}
        batch_pass = 0
        max_id = 0
        center = dataset_dict[0]['center_meter']
        resolution = np.array(dataset_dict[0]['bbox_pixel']) / np.array(
            dataset_dict[0]['bbox_meter'])
        plot_data = {
            "pred": {
                "velocity": [],
                "yaw_rate": [],
                "acceleration": [],
                "position_diff": [],
            },
            "gt": {
                "velocity": [],
                "yaw_rate": [],
                "acceleration": [],
            },
            "meta": {
                "goal_points": [],
            }
        }
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            if batch_idx > 566600:
                break
            x_image = sample['images']
            x_image = x_image.to(device)
            x_traj = [sample['hist_objs'].to(device), sample['hist_obj_lens']]
            x_traj_len = sample['hist_objs_seq_len']
            x_traj_pred_obj_len = sample['pred_obj_lens']
            x_traj_pred_len = sample['pred_objs_seq_len']
            pres_object_lengths_sum = sample['pres_object_lengths_sum']
            obj_length_padded = sample['hist_object_lengths_sum']
            batch_wise_decoder_input = sample['obj_decoder_in'].to(device)
            gTruthX = sample['pred_objsx'].to(device)
            gTruthY = sample['pred_objsy'].to(device)
            gTruthT = sample['pred_objst'].to(device, non_blocking=True)

            target_len = 30
            # frame = 2
            x_offset = [0, 0]
            y_offset = [0, 0]
            vehicle_idx = [2, 3]
            frame = frame_idx


            # sample["cond_goal_point"][frame][3][0] = sample["cond_goal_point"][frame][3][0] + 0
            # sample["cond_goal_point"][frame][3][1] = sample["cond_goal_point"][frame][3][1] - 0

            condition_goal_point = sample['cond_goal_point'].to(device, non_blocking=True).float().to(
                device, non_blocking=True)
            condition_v = sample['cond_v'].to(device, non_blocking=True).float().to(device,
                                                                                    non_blocking=True)
            combined_condition = torch.cat([condition_goal_point], dim=2)

            output = model(x_image=x_image, x_traj=x_traj, x_traj_len=x_traj_len,
                           batch_wise_object_lengths_sum=obj_length_padded,
                           conditions=combined_condition,
                           batch_wise_decoder_input=batch_wise_decoder_input, target_length=target_len)
            X = output['X']
            Y = output['Y']
            T = output['T']
            X_reshaped, Y_reshaped, T_reshaped = eval_730._process_dynamic_model(
                output, gTruthX, gTruthY, gTruthT, x_traj_pred_obj_len,
                pres_object_lengths_sum)
            gt_v = sample["pred_objsv"]
            gt_a_lat = sample["pred_objs_a_lat"]
            gt_a_lon = sample["pred_objs_a_lon"]
            gt_psi = sample["pred_objspsi"]
            gt_v, gt_a_lon, gt_a_lat = eval_730._process_dynamic_model_reshape(output, gt_v, gt_a_lon, gt_a_lat,
                                                                               x_traj_pred_obj_len,
                                                                               pres_object_lengths_sum)
            for idx, image in enumerate(x_image):
                if idx != frame:
                    continue
                cond_acc = sample['cond_acc'][idx].cpu().numpy()
                cond_goal_point = sample['cond_goal_point'][idx].cpu().numpy()
                image = image[0, :, :].cpu().numpy()
                psi_dots = output['psi_dot'][obj_length_padded[idx]:obj_length_padded[idx + 1], :].cpu().numpy()

                # if psi_dots.max() < 10:
                #     continue
                ax = output['ax'][obj_length_padded[idx]:obj_length_padded[idx + 1], :].cpu().numpy()
                v = output['v'][obj_length_padded[idx]:obj_length_padded[idx + 1], :].cpu().numpy()

                # Define a list of colors for the predictions
                prediction_colors = [
                    (0, 0, 1),  # Blue
                    (1, 0, 0),  # Red
                    (0, 1, 0),  # Green
                    (0, 1, 1),  # Cyan
                    (1, 0, 1),  # Magenta
                    (0.5, 0.5, 0),  # Olive
                    (0.5, 0, 0.5),  # Purple
                    (0.5, 0, 0),  # Maroon
                    (0.75, 0.75, 0),  # Light Olive
                    (0.75, 0, 0.75),  # Light Magenta
                    (0, 0.5, 0.5),  # Teal
                    (0.75, 0.25, 0),  # Orange
                    (0.25, 0.75, 0),  # Lime
                    (1, 1, 0),  # Yellow
                    (0, 0.25, 0.75)  # Royal Blue
                ]

                # Create a figure with a 2x2 grid layout

                traj_hist = x_traj[0][idx, :, :, :].cpu().numpy()
                gTruthX_plot = gTruthX[idx, :, :, :].cpu().numpy()
                gTruthY_plot = gTruthY[idx, :, :, :].cpu().numpy()
                X_reshaped_plot = X_reshaped[idx, :, :, :].cpu().numpy()
                Y_reshaped_plot = Y_reshaped[idx, :, :, :].cpu().numpy()
                gt_v = gt_v[idx, :, :, :].cpu().numpy()
                gt_a_lat = gt_a_lat[idx, :, :, :].cpu().numpy()
                gt_a_lon = gt_a_lon[idx, :, :, :].cpu().numpy()
                gt_psi = gt_psi[idx, :, :, :].cpu().numpy()
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


                x_traj_len_local = x_traj_len[obj_length_padded[idx]:obj_length_padded[idx + 1]]
                x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx]:obj_length_padded[idx + 1]]
                cmap_objec = get_cmap(X_reshaped_plot.shape[0])
                plot_v = []
                plot_a_lon = []
                plot_a_lat = []
                plot_psi_gt = []
                ade_diff = []
                for id, (x, y, x_temp_hist, y_temp_hist, gt_x, gt_y, gt_v_, gt_a_lat_, gt_a_lon_, gt_psi_) in \
                        enumerate(zip(X_reshaped_plot[0:x_traj[1][idx], :, :],
                                      Y_reshaped_plot[0:x_traj[1][idx], :, :],
                                      traj_hist[0:x_traj[1][idx], :, 0],
                                      traj_hist[0:x_traj[1][idx], :, 1],
                                      gTruthX_plot[0:x_traj[1][idx], :, :],
                                      gTruthY_plot[0:x_traj[1][idx], :, :],
                                      gt_v[0:x_traj[1][idx], :, :],
                                      gt_a_lat[0:x_traj[1][idx], :, :],
                                      gt_a_lon[0:x_traj[1][idx], :, :],
                                      gt_psi[0:x_traj[1][idx], :, :])):
                    # Prediction
                    x = x[:x_traj_pred_len_local[id], :]
                    y = y[:x_traj_pred_len_local[id], :]
                    gt_v_ = gt_v_[:x_traj_pred_len_local[id], :]
                    plot_v.append(gt_v_)
                    plot_a_lon.append(gt_a_lon_)
                    plot_a_lat.append(gt_a_lat_)
                    plot_psi_gt.append(gt_psi_)
                    # History
                    x_temp_hist = x_temp_hist[:x_traj_len_local[id], ]
                    y_temp_hist = y_temp_hist[:x_traj_len_local[id], ]
                    gt_x = gt_x[:x_traj_pred_len_local[id], :]
                    gt_y = gt_y[:x_traj_pred_len_local[id], :]

                    points = np.hstack((x, y))  # Shape: (30, 2)
                    gt_points = np.hstack((gt_x, gt_y))  # Shape: (30, 2)

                    # Compute the Euclidean distance for each timestep
                    distances = np.linalg.norm(points - gt_points, axis=1)
                    ade_diff.append(distances)

                    # Plot historical trajectory
                    line_cor = [
                        (int(x_temp), int(y_temp))
                        for x_temp, y_temp in
                        zip((x_temp_hist + center[0]) * resolution[0],
                            dataset_dict[0]['bbox_pixel'][1] -
                            (y_temp_hist + center[1]) * resolution[1])
                    ]
                    line_cor = np.array(line_cor).reshape(-1, 1, 2)
                    image = cv2.polylines(image, [line_cor], False, cmap_objec(id), 2)

                    # Plot prediction trajectory with unique color
                    prediction_color = prediction_colors[id % len(prediction_colors)]
                    line_cor = [
                        (int(x_temp), int(y_temp))
                        for x_temp, y_temp in
                        zip((x + center[0]) * resolution[0],
                            dataset_dict[0]['bbox_pixel'][1] -
                            (y + center[1]) * resolution[1])
                    ]
                    line_cor = np.array(line_cor).reshape(-1, 1, 2)
                    image = cv2.polylines(image, [line_cor], False, prediction_color, 1)

                # Initialize storage containers


                # Collect predicted velocities
                for i in range(v.shape[0]):
                    valid_len = x_traj_pred_len_local[i]
                    plot_data["pred"]["velocity"].append(v[i, :valid_len].tolist())

                # Collect predicted yaw rates
                for i in range(psi_dots.shape[0]):
                    valid_len = x_traj_pred_len_local[i]
                    plot_data["pred"]["yaw_rate"].append(psi_dots[i, :valid_len].tolist())

                # Collect predicted accelerations
                for i in range(ax.shape[0]):
                    valid_len = x_traj_pred_len_local[i]
                    plot_data["pred"]["acceleration"].append(ax[i, :valid_len].tolist())

                # Store goal points (as pixel coordinates after conversion)
                goal_points_pixel = []
                for idx, pos in enumerate(cond_goal_point):
                    if pos[0] == 0.0 and pos[1] == 0.0:
                        continue
                    x = int((pos[0] + center[0]) * resolution[0])
                    y = int(dataset_dict[0]['bbox_pixel'][1] - (pos[1] + center[1]) * resolution[1])
                    if idx in vehicle_idx:
                        x_offset_i = x_offset[vehicle_idx.index(idx)]
                        y_offset_i = y_offset[vehicle_idx.index(idx)]
                        x += x_offset_i
                        y += y_offset_i
                    goal_points_pixel.append((x, y))
                plot_data["meta"]["goal_points"] = goal_points_pixel

                # Collect difference to ground truth
                for i in range(len(ade_diff)):
                    valid_len = x_traj_pred_len_local[i]
                    plot_data["pred"]["position_diff"].append(ade_diff[i][:valid_len].tolist())

                # Ground truth velocity
                # Ground truth velocity
                for tensor in plot_v:
                    cleaned = np.array(tensor)
                    cleaned = cleaned[~np.isnan(cleaned)]
                    plot_data["gt"]["velocity"].append(cleaned.tolist())

                # Ground truth yaw rate (computed from Psi using np.gradient)
                plot_psi_dot_gt = [np.gradient(tensor.squeeze(), 0.1) for tensor in plot_psi_gt]
                for tensor in plot_psi_dot_gt:
                    cleaned = np.array(tensor)
                    cleaned = cleaned[~np.isnan(cleaned)]
                    plot_data["gt"]["yaw_rate"].append(cleaned.tolist())

                # Ground truth longitudinal acceleration
                plot_a_lon = torch.tensor(plot_a_lon)
                for i in range(plot_a_lon.shape[0]):
                    cleaned = plot_a_lon[i]
                    cleaned = cleaned[~torch.isnan(cleaned)]
                    plot_data["gt"]["acceleration"].append(cleaned.tolist())

        def flatten_no_nan(data):
            flat = [val for sublist in data for val in sublist]
            return np.array([v for v in flat if True])

        def compute_abs_mean_error(gt_list, pred_list):
            """Compute mean absolute error per trajectory using pure Python."""
            errors = []
            for gt_seq, pred_seq in zip(gt_list, pred_list):
                if len(gt_seq) != len(pred_seq) or len(gt_seq) == 0:
                    continue  # skip mismatched or empty sequences

                abs_errors = [g - p for g, p in zip(gt_seq, pred_seq)]
                mean_error = sum(abs_errors) / len(abs_errors)
                errors.append(mean_error)
            return errors

        def plot_all_errors(gt_data_dict, pred_data_dict):
            with open("./gt_data_dict", "wb") as f:
                pickle.dump(gt_data_dict, f)
            with open("./pred_data_dict", "wb") as f:
                pickle.dump(pred_data_dict, f)
            gt_ax = gt_data_dict["acceleration"]
            pred_ax = pred_data_dict["acceleration"]

            gt_psi_dot = gt_data_dict["yaw_rate"]
            pred_psi_dot = pred_data_dict["yaw_rate"]

            gt_v = gt_data_dict["velocity"]
            pred_v = pred_data_dict["velocity"]

            # Clip yaw rate values in-place
            for i in range(len(gt_psi_dot)):
                gt_psi_dot[i] = [max(min(val, 3.14), -3.14) for val in gt_psi_dot[i]]
                pred_psi_dot[i] = [max(min(val, 3.14), -3.14) for val in pred_psi_dot[i]]

            # Compute errors
            ax_errors = compute_abs_mean_error(gt_ax, pred_ax)
            psi_dot_errors = compute_abs_mean_error(gt_psi_dot, pred_psi_dot)
            v_errors = compute_abs_mean_error(gt_v, pred_v)

            # Plot histograms of errors
            fig, axes = plt.subplots(2, 1, figsize=(18, 12))
            titles = ["Acceleration Error", "Yaw Rate Error"]
            xlabels = ["Mean Abs Error [m/s²]", "Mean Abs Error [rad/s]"]
            error_lists = [ax_errors, psi_dot_errors, v_errors]

            for i, (title, xlabel, errors) in enumerate(zip(titles, xlabels, error_lists)):
                axes[i].hist(errors, bins=50, alpha=0.75, color='tab:blue')
                axes[i].set_title(title)
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel("Trajectory Count")
                axes[i].grid(True)

            plt.tight_layout()
            plt.show()
        # Call the function
        plot_all_errors(plot_data["gt"], plot_data["pred"])
