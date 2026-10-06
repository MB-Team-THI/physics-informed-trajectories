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
from src.utils.get_ade import get_ade, get_displacement_per_timestep
import cv2
from src.utils.rot_points import rot_points
from einops import rearrange
from src.utils.get_cmap import get_cmap
import logging
import matplotlib.pyplot as plt
from src.utils.get_img_from_fig import get_img_from_fig
import os
import pandas as pd
import seaborn as sns
import imageio
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

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

class eval_751(eval):

    def __init__(self,
                 idx=121,
                 name='Interactive scenario evaluation',
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

    def _evaluate(self, model, dataloader_test, device, dataset_dict, frame_idx=0):
        """
        Main evaluation function, but implemented to allow interactive picking of a scenario & vehicle.
        Then the user can click to set a new goal point, which re-runs the model and updates the figure.
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()
        if next(model.parameters()).device is not device:
            model.to(device)

        # Convert the dataloader into a list (or just one big list if it isn't too large),
        # so we can pick scenarios easily. In practice, you might store only relevant batches
        gt_psi_list  = []
        gt_ax_list  = []
        pred_psi_list  = []
        pred_ax_list  = []

        control_inputs_ax = []
        control_inputs_psi = []

        ade_timestep_list = []
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            if batch_idx > 100:
                break

            # Let the user pick the single vehicle index that we will manipulate
            # We'll re-run the model each time the user clicks.
            assert sample['pred_objsx'].shape[0] == 1, "This example assumes batch size 1"

            self.sample_cache = sample


            # Prepare sample for inference
            x_image = self.sample_cache['images'].to(self.device)
            x_traj = [self.sample_cache['hist_objs'].to(self.device), self.sample_cache['hist_obj_lens']]
            x_traj_len = self.sample_cache['hist_objs_seq_len']
            x_traj_pred_obj_len = self.sample_cache['pred_obj_lens']
            x_traj_pred_len = self.sample_cache['pred_objs_seq_len']
            pres_object_lengths_sum = self.sample_cache['pres_object_lengths_sum']
            obj_length_padded = self.sample_cache['hist_object_lengths_sum']

            batch_wise_decoder_input = self.sample_cache['obj_decoder_in'].to(self.device)
            gTruthX = self.sample_cache['pred_objsx'].to(self.device)
            gTruthY = self.sample_cache['pred_objsy'].to(self.device)
            gTruthT = self.sample_cache['pred_objst'].to(self.device, non_blocking=True)

            # Ground-truth for velocity, acceleration, yaw
            gt_v = self.sample_cache['pred_objsv'].to(self.device)
            gt_a_lat = self.sample_cache['pred_objs_a_lat'].to(self.device)
            gt_a_lon = self.sample_cache['pred_objs_a_lon'].to(self.device)
            gt_psi = self.sample_cache['pred_objspsi'].to(self.device)

            condition_goal_point = self.sample_cache['cond_goal_point'].to(self.device, non_blocking=True).float()
            condition_v = self.sample_cache['cond_v'].to(self.device, non_blocking=True).float()
            combined_condition = torch.cat([condition_goal_point], dim=2)

            with torch.no_grad():
                output = self.model(
                    x_image=x_image,
                    x_traj=x_traj,
                    x_traj_len=x_traj_len,
                    batch_wise_object_lengths_sum=obj_length_padded,
                    conditions=combined_condition,
                    batch_wise_decoder_input=batch_wise_decoder_input,
                    target_length=self.target_len
                )

            # Reshape predictions
            X_reshaped, Y_reshaped, T_reshaped = self._process_dynamic_model(
                output, gTruthX, gTruthY, gTruthT,
                x_traj_pred_obj_len, pres_object_lengths_sum
            )

            gt_v_reshaped, gt_a_lon_reshaped, gt_a_lat_reshaped = eval_730._process_dynamic_model_reshape(
                output, gt_v, gt_a_lon, gt_a_lat,
                x_traj_pred_obj_len, pres_object_lengths_sum
            )
            dummyX = torch.zeros_like(gt_psi, device=gt_psi.device)
            dummyY = torch.zeros_like(gt_psi, device=gt_psi.device)
            gt_psi_reshaped, _, _ = eval_730._process_dynamic_model_reshape(
                output, gt_psi, dummyX, dummyY,
                x_traj_pred_obj_len, pres_object_lengths_sum
            )

            def compute_smoothness_loss(signal, valid_lens):
                # signal: (n, T)
                n, T = signal.shape
                total_loss = 0.0
                count = 0

                for i in range(n):
                    valid_len = valid_lens[i]

                    # Need at least 2 timesteps to compute 1 difference
                    if valid_len > 1:
                        # Get valid part of the signal for this sample
                        valid_signal = signal[i, :valid_len]
                        differences = valid_signal[1:] - valid_signal[:-1]  # shape: (valid_len - 1,)
                        loss = torch.mean(differences ** 2)
                        total_loss += loss
                        count += 1

                if count > 0:
                    return total_loss / count
                else:
                    return torch.tensor(0.0, device=signal.device)
            valid = x_traj_pred_len
            gt_a_lon_reshaped = gt_a_lon_reshaped.squeeze() # n, 30

            try:
                gt_psi_reshaped = gt_psi_reshaped.squeeze() # n, 30
                if len(gt_psi_reshaped.shape )== 1:
                    gt_psi_reshaped = gt_psi_reshaped.unsqueeze(0)
                    gt_a_lon_reshaped = gt_a_lon_reshaped.unsqueeze(0)


                gt_psi_reshaped = (gt_psi_reshaped[:, 1:] - gt_psi_reshaped[:, :-1]) / 0.1  # shape: (n, T-1)
                gt_psi_reshaped = F.pad(gt_psi_reshaped, (1, 0), mode='replicate')  # shape: (n, T)
            except RuntimeError as e:
                print(f"Error in gt_psi_reshaped: {e}")
                gt_psi_reshaped = torch.zeros_like(gt_psi_reshaped)

            pred_a_lon = output["ax"]
            pred_psi_dot = output["psi_dot"]



            gt_smoothness_loss_psi = compute_smoothness_loss(gt_psi_reshaped, valid)
            gt_smoothness_loss_ax = compute_smoothness_loss(gt_a_lon_reshaped, valid)

            pred_smoothness_loss_psi = compute_smoothness_loss(pred_psi_dot, valid)
            pred_smoothness_loss_ax = compute_smoothness_loss(pred_a_lon, valid)
            control_inputs_psi.append(pred_psi_dot)
            control_inputs_ax.append(pred_a_lon)
            gt_psi_list.append(gt_smoothness_loss_psi.item())
            gt_ax_list.append(gt_smoothness_loss_ax.item())
            pred_psi_list.append(pred_smoothness_loss_psi.item())
            pred_ax_list.append(pred_smoothness_loss_ax.item())



        gt_psi_array = np.array(gt_psi_list)
        gt_ax_array = np.array(gt_ax_list)
        pred_psi_array = np.array(pred_psi_list)
        pred_ax_array = np.array(pred_ax_list)

        control_inputs_psi = torch.cat(control_inputs_psi, dim=0).cpu().numpy()
        control_inputs_ax = torch.cat(control_inputs_ax, dim=0).cpu().numpy()

        # # Flatten the arrays
        # psi_flat = control_inputs_psi.flatten()
        # ax_flat = control_inputs_ax.flatten()
        #
        # # Define number of bins
        # num_bins = 512 // 10
        #
        # # Compute histograms
        # psi_counts, psi_bins = np.histogram(psi_flat, bins=num_bins)
        # ax_counts, ax_bins = np.histogram(ax_flat, bins=num_bins)
        #
        # # Plotting
        # plt.figure(figsize=(14, 6))
        #
        # # Bar plot for psi
        # plt.subplot(1, 2, 1)
        # plt.bar(psi_bins[:-1], psi_counts, width=np.diff(psi_bins), align='edge', color='tab:blue')
        # plt.title("Distribution of control_inputs_psi (ψ̇)")
        # plt.xlabel("ψ̇ value")
        # plt.ylabel("Frequency")
        #
        # # Bar plot for ax
        # plt.subplot(1, 2, 2)
        # plt.bar(ax_bins[:-1], ax_counts, width=np.diff(ax_bins), align='edge', color='tab:orange')
        # plt.title("Distribution of control_inputs_ax (aₓ)")
        # plt.xlabel("aₓ value")
        # plt.ylabel("Frequency")
        #
        # plt.tight_layout()
        # plt.show()






        np.save("./gt_psi_array.npy", gt_psi_array)
        np.save("./gt_ax_array.npy", gt_ax_array)
        np.save("./pred_psi_array.npy", pred_psi_array)
        np.save("./pred_ax_array.npy", pred_ax_array)

        np.save("./control_inputs_psi.npy", control_inputs_psi)
        np.save("./control_inputs_ax.npy", control_inputs_ax)

        gt_psi_mean = np.mean(gt_psi_array)
        gt_ax_mean = np.mean(gt_ax_array)
        pred_psi_mean = np.mean(pred_psi_array)
        pred_ax_mean = np.mean(pred_ax_array)

        print("GT psi mean: ", gt_psi_mean)
        print("GT ax mean: ", gt_ax_mean)
        print("Pred psi mean: ", pred_psi_mean)
        print("Pred ax mean: ", pred_ax_mean)

