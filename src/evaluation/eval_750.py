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

class eval_750(eval):

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
        fde_list  = []
        ade_list = []
        ade_timestep_list = []
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            if batch_idx > 500:
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

            maskXnot = torch.isnan(gTruthX)
            maskYnot = torch.isnan(gTruthX)

            X_mse = F.mse_loss(X_reshaped[~maskXnot], gTruthX[~maskXnot])
            Y_mse = F.mse_loss(Y_reshaped[~maskYnot], gTruthY[~maskYnot])
            with torch.no_grad():
                ade = get_ade(X_reshaped, Y_reshaped, gTruthX, gTruthY,
                              maskXnot, maskYnot)
                ade_timestep = get_displacement_per_timestep(X_reshaped, Y_reshaped, gTruthX, gTruthY,
                              maskXnot, maskYnot)
                ade_timestep_list.append(ade_timestep.cpu())
                fde = get_fde(X_reshaped, Y_reshaped, gTruthX, gTruthY,
                              x_traj_pred_len, maskXnot, maskXnot)
                ade_list.append(ade.cpu().item())
                fde_list.append(fde.cpu().item())

        ade_array = np.array(ade_list)
        fde_array = np.array(fde_list)

        np.save("./ade_array.npy", ade_array)
        np.save("./fde_array.npy", fde_array)


        # Compute mean values
        ade_mean = np.mean(ade_array)
        fde_mean = np.mean(fde_array)

        ade_all_tensor = torch.cat(ade_timestep_list, dim=1).squeeze(0) # shape: (B, n, 30)
        ade_avg_per_timestep = torch.nanmean(ade_all_tensor, dim=0)  # shape: (30,)
        ade_np = ade_avg_per_timestep.cpu().numpy()
        # Create a DataFrame for seaborn
        df = pd.DataFrame({
            'Timestep': range(len(ade_np)),
            'ADE': ade_np
        })



        # Plot using seaborn
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x='Timestep', y='ADE', marker='o')
        plt.title('Average Displacement Error per Timestep')
        plt.xlabel('Timestep')
        plt.ylabel('ADE')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Set Seaborn style
        sns.set(style="whitegrid")

        # Create the figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # ADE Distribution Plot
        sns.histplot(ade_array, bins=20, kde=False, ax=axs[0], edgecolor='black', color='skyblue')
        axs[0].axvline(ade_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {ade_mean:.3f}')
        axs[0].set_xlabel('ADE')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('ADE Distribution')
        axs[0].legend()

        # FDE Distribution Plot
        sns.histplot(fde_array, bins=20, kde=False, ax=axs[1], edgecolor='black', color='lightgreen')
        axs[1].axvline(fde_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {fde_mean:.3f}')
        axs[1].set_xlabel('FDE')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('FDE Distribution')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

