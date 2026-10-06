import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.widgets import RadioButtons, Slider
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
                 name='MSE prediction accuracy',
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

class eval_742(eval):
    """
    This class extends the functionality of eval_730 by allowing a user to interactively choose:
      - A scenario (batch index)
      - A vehicle index
      - And then click on a plot to set a new goal point for that vehicle.

    After setting a new goal point, the model will re-run and update predictions on the fly.
    We keep the interface with __call__(...) and name it eval_740.

    Usage:
        e = eval_740(...)
        e(model=..., dataloader_test=..., device=..., dataset_dict=...)

    Once _evaluate(...) is called, a small interactive window with matplotlib will appear.
    The user will:
      1) Input scenario index from the console
      2) Input vehicle index from the console
      3) On the displayed image, a user can click to set a new goal point
      4) The model re-runs, predictions are updated, and the figure is re-drawn
      5) The user can continue clicking new points. Close the figure to end the loop.

    This is a simple demonstration of how to add interactive steps.
    A more advanced approach can use frameworks like Gradio, Plotly Dash, or Streamlit for a web UI.
    """

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
        self.temperature = 1.0

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
        all_samples = []
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            all_samples.append(sample)
            if batch_idx > 50: break
        if not all_samples:
            print("No data in dataloader. Exiting.")
            return None

        # Let the user pick a scenario index from the console
        # (In practice, you could build a more sophisticated UI)
        max_scenario_idx = len(all_samples)
        scenario_idx = "9"
        # scenario_idx = input(
        #     f"Enter scenario index [1..{max_scenario_idx}] (default=0): "
        # )
        if scenario_idx.strip() == "":
            scenario_idx = 0
        else:
            scenario_idx = int(scenario_idx) - 1
        if scenario_idx > max_scenario_idx or scenario_idx < 0:
            scenario_idx = 0

        self.current_scenario_idx = scenario_idx
        # We only take that one scenario (batch).
        sample = all_samples[scenario_idx]

        # Let the user pick the single vehicle index that we will manipulate
        # We'll re-run the model each time the user clicks.
        max_vehicle_idx = 15
        vehicle_idx = "0"
        if vehicle_idx.strip() == "":
            vehicle_idx = 0
        else:
            vehicle_idx = int(vehicle_idx) -1
        if vehicle_idx > max_vehicle_idx or vehicle_idx < 0:
            vehicle_idx = 0

        self.current_vehicle_idx = vehicle_idx
        self.sample_cache = sample

        def update_vehicle(label):
            """
            Callback when the user selects a new vehicle from the dropdown.
            """
            new_vehicle_idx = int(label) -1
            self.current_vehicle_idx = new_vehicle_idx
            print(f"Vehicle changed to: {self.current_vehicle_idx}")

            # Re-run inference and update the plot with the new vehicle
            run_inference_and_plot()

        # We'll define a helper to run the model and plot results
        def run_inference_and_plot(new_goal_x=None, new_goal_y=None):
            """
            1) Modify the sample's goal point for the chosen vehicle.
            2) Run the model forward.
            3) Plot the result in self.fig / self.axs (now with a 3x3 grid).
            """

            # If the user clicked, set the new goal point in cond_goal_point
            if new_goal_x is not None and new_goal_y is not None:
                # Overwrite the relevant vehicle's goal for this scenario
                self.sample_cache['cond_goal_point'][frame_idx, self.current_vehicle_idx, 0] = new_goal_x
                self.sample_cache['cond_goal_point'][frame_idx, self.current_vehicle_idx, 1] = new_goal_y

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
                    target_length=self.target_len,
                    temperature=self.temperature,
                )

            # Reshape predictions
            X_reshaped, Y_reshaped, T_reshaped = self._process_dynamic_model(
                output, gTruthX, gTruthY, gTruthT,
                x_traj_pred_obj_len, pres_object_lengths_sum
            )

            # Also reshape ground truth arrays for velocity/accel/yaw so we can index them consistently
            # We'll use the same "3-tensor" reshape helper on each triple as needed.
            # If you need them individually, define a separate method or do them one by one.
            gt_v_reshaped, gt_a_lon_reshaped, gt_a_lat_reshaped = eval_730._process_dynamic_model_reshape(
                output, gt_v, gt_a_lon, gt_a_lat,
                x_traj_pred_obj_len, pres_object_lengths_sum
            )
            # If you want to reshape psi as well, do a separate call that aligns shapes:
            # (We can do a single triple if your _process_dynamic_model_reshape can handle it).
            # For simplicity, just do a smaller function or a second triple call if needed:
            # e.g.   gt_psi_reshaped, dummy1, dummy2 = ...
            # but we only need gt_psi reshaped as a single "X" and the other 2 placeholders "Y, T".
            # We'll just do a quick approach by reusing the same function:
            dummyX = torch.zeros_like(gt_psi, device=gt_psi.device)
            dummyY = torch.zeros_like(gt_psi, device=gt_psi.device)
            gt_psi_reshaped, _, _ = eval_730._process_dynamic_model_reshape(
                output, gt_psi, dummyX, dummyY,
                x_traj_pred_obj_len, pres_object_lengths_sum
            )

            idx = frame_idx  # We only show the frame of interest
            center = self.dataset_dict[0]['center_meter']
            resolution = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            H_img = self.dataset_dict[0]['bbox_pixel'][1]

            # Grab relevant slices for plotting
            image = x_image[idx, 0, :, :].cpu().numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # These are per-batch sets of indices
            x_traj_len_local = x_traj_len[obj_length_padded[idx]: obj_length_padded[idx + 1]]
            x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx]: obj_length_padded[idx + 1]]

            # Model outputs per-object
            psi_dots = output['psi_dot'][obj_length_padded[idx]: obj_length_padded[idx + 1], :].cpu().numpy()
            ax = output['ax'][obj_length_padded[idx]: obj_length_padded[idx + 1], :].cpu().numpy()
            v = output['v'][obj_length_padded[idx]: obj_length_padded[idx + 1], :].cpu().numpy()

            # Historical & predicted positions
            traj_hist = x_traj[0][idx].cpu().numpy()  # [num_vehicles, hist_length, 3] (x,y,theta) or similar
            X_reshaped_plot = X_reshaped[idx].cpu().numpy()  # [num_vehicles, pred_length, 1]
            Y_reshaped_plot = Y_reshaped[idx].cpu().numpy()  # [num_vehicles, pred_length, 1]
            gTruthX_plot = gTruthX[idx].cpu().numpy()
            gTruthY_plot = gTruthY[idx].cpu().numpy()

            # Ground-truth expansions (already reshaped)
            gt_v_plot = gt_v_reshaped[idx][obj_length_padded[idx]: obj_length_padded[idx + 1]].cpu().numpy()  # [num_vehicles, pred_length, 1]
            gt_a_lat_plot = gt_a_lat_reshaped[idx][obj_length_padded[idx]: obj_length_padded[idx + 1]].cpu().numpy()
            gt_a_lon_plot = gt_a_lon_reshaped[idx][obj_length_padded[idx]: obj_length_padded[idx + 1]].cpu().numpy()
            gt_psi_plot = gt_psi_reshaped[idx][obj_length_padded[idx]: obj_length_padded[idx + 1]].cpu().numpy()

            # Define a list of colors for each predicted trajectory
            prediction_colors = [
                (0, 0, 1),  # Blue
                (1, 0, 0),  # Red
                (0, 1, 0),  # Green
                (0, 1, 1),  # Cyan
                (1, 0, 1),  # Magenta
                (0.5, 0.5, 0),
                (0.5, 0, 0.5),
                (0.5, 0, 0),
                (0.75, 0.75, 0),
                (0.75, 0, 0.75),
                (0, 0.5, 0.5),
                (0.75, 0.25, 0),
                (0.25, 0.75, 0),
                (1, 1, 0),
                (0, 0.25, 0.75)
            ]

            # Create a fresh 3x3 figure if we don't have one yet
            if self.fig is None or self.axs is None:
                self.fig, self.axs = plt.subplots(3, 3, figsize=(16, 12))
            else:
                # Clear old content
                for row_ax in self.axs:
                    for ax_item in row_ax:
                        ax_item.clear()

            # =====================
            # (A) Trajectory Visualization: [0, 0]
            # =====================
            # Plot historical + predicted polylines into image_rgb
            ade_diff = []  # store difference to ground truth per vehicle
            for veh_id in range(x_traj[1][idx]):  # x_traj[1][idx] = number_of_vehicles in this scenario
                # History
                h_len = x_traj_len_local[veh_id]
                x_hist = traj_hist[veh_id, :h_len, 0]
                y_hist = traj_hist[veh_id, :h_len, 1]

                # Prediction
                p_len = x_traj_pred_len_local[veh_id]
                pred_x = X_reshaped_plot[veh_id, :p_len, 0]
                pred_y = Y_reshaped_plot[veh_id, :p_len, 0]

                # Ground truth
                gt_x = gTruthX_plot[veh_id, :p_len, 0]
                gt_y = gTruthY_plot[veh_id, :p_len, 0]

                # Accumulate position error for each time step
                pred_pts = np.stack([pred_x, pred_y], axis=1)  # shape [p_len, 2]
                gt_pts = np.stack([gt_x, gt_y], axis=1)  # shape [p_len, 2]
                dist = np.linalg.norm(pred_pts - gt_pts, axis=1)  # shape [p_len]
                ade_diff.append(dist)
                # Convert predicted coords & draw
                pred_color = prediction_colors[veh_id % len(prediction_colors)]

                # Convert history to pixel coords & draw
                hist_coords = []
                for (xh, yh) in zip(x_hist, y_hist):
                    px = int((xh + center[0]) * resolution[0])
                    py = int(H_img - (yh + center[1]) * resolution[1])
                    hist_coords.append((px, py))
                hist_coords = np.array(hist_coords).reshape(-1, 1, 2)
                image_rgb = cv2.polylines(image_rgb, [hist_coords.astype(np.int32)],
                                          False, pred_color, 1)


                pred_coords = []
                for (px, py) in zip(pred_x, pred_y):
                    pixx = int((px + center[0]) * resolution[0])
                    pixy = int(H_img - (py + center[1]) * resolution[1])
                    pred_coords.append((pixx, pixy))
                pred_coords = np.array(pred_coords).reshape(-1, 1, 2)
                image_rgb = cv2.polylines(image_rgb, [pred_coords.astype(np.int32)],
                                          False, pred_color, 2)

            # Plot the goal points as cross markers
            cond_goal = self.sample_cache['cond_goal_point'][idx].cpu().numpy()
            for v_idx in range(cond_goal.shape[0]):
                gx = cond_goal[v_idx, 0]
                gy = cond_goal[v_idx, 1]
                if (gx == 0.0 and gy == 0.0):
                    continue
                x_pix = int((gx + center[0]) * resolution[0])
                y_pix = int(H_img - (gy + center[1]) * resolution[1])
                cv2.drawMarker(image_rgb, (x_pix, y_pix), (255, 0, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            # Show the image on [0,0]
            self.axs[0, 0].imshow(image_rgb)
            self.axs[0, 0].axis("off")
            self.axs[0, 0].set_title(
                f"Trajectories (Scenario={self.current_scenario_idx}, Veh={self.current_vehicle_idx})")

            # =====================
            # (B) Predicted Velocity: [0, 1]
            # =====================
            for i in range(v.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                self.axs[0, 1].plot(
                    v[i, :valid_len], marker='o',
                    label=f'Veh {i + 1}',
                    color=color
                )
            self.axs[0, 1].set_xlabel('Time Step')
            self.axs[0, 1].set_ylabel('Velocity [m/s]')
            self.axs[0, 1].set_title('Predicted Velocity')
            self.axs[0, 1].legend()
            self.axs[0, 1].grid(True)

            # =====================
            # (C) Ground-Truth Velocity: [0, 2]
            # =====================
            for i in range(gt_v_plot.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                # gt_v_plot[i, :valid_len, 0] is your ground-truth velocity
                self.axs[0, 2].plot(
                    gt_v_plot[i, :valid_len, 0],
                    marker='o',
                    label=f'Veh {i + 1}',
                    color=color
                )
            self.axs[0, 2].set_xlabel('Time Step')
            self.axs[0, 2].set_ylabel('Velocity [m/s]')
            self.axs[0, 2].set_title('Ground Truth Velocity')
            self.axs[0, 2].legend()
            self.axs[0, 2].grid(True)

            # =====================
            # (D) Position Difference to Ground Truth: [1, 0]
            # =====================
            for i, dist_array in enumerate(ade_diff):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                self.axs[1, 0].plot(dist_array[:valid_len], marker='o',
                                    label=f'Veh {i + 1}', color=color)
            self.axs[1, 0].set_xlabel('Time Step')
            self.axs[1, 0].set_ylabel('Error [m]')
            self.axs[1, 0].set_title('Position Error to Ground Truth')
            self.axs[1, 0].legend()
            self.axs[1, 0].grid(True)

            # =====================
            # (E) Predicted Longitudinal Acceleration: [1, 1]
            # =====================
            for i in range(ax.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                self.axs[1, 1].plot(ax[i, :valid_len], marker='o',
                                    label=f'Veh {i + 1}', color=color)
            self.axs[1, 1].set_xlabel('Time Step')
            self.axs[1, 1].set_ylabel('Accel [m/s^2]')
            self.axs[1, 1].set_title('Predicted Longitudinal Acceleration')
            self.axs[1, 1].legend()
            self.axs[1, 1].grid(True)

            # =====================
            # (F) Ground-Truth Longitudinal Acceleration: [1, 2]
            # =====================
            for i in range(gt_a_lon_plot.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                # gt_a_lon_plot[i, :valid_len, 0] is your GT lon accel
                self.axs[1, 2].plot(
                    gt_a_lon_plot[i, :valid_len, 0],
                    marker='o',
                    label=f'Veh {i + 1}',
                    color=color
                )
            self.axs[1, 2].set_xlabel('Time Step')
            self.axs[1, 2].set_ylabel('Accel [m/s^2]')
            self.axs[1, 2].set_title('Ground Truth Longitudinal Acceleration')
            self.axs[1, 2].legend()
            self.axs[1, 2].grid(True)

            # =====================
            # (G) Predicted Yaw Rate: [2, 1]
            # =====================
            for i in range(psi_dots.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                self.axs[2, 1].plot(psi_dots[i, :valid_len], marker='o',
                                    label=f'Veh {i + 1}', color=color)
            self.axs[2, 1].set_xlabel('Time Step')
            self.axs[2, 1].set_ylabel('Yaw Rate [rad/s]')
            self.axs[2, 1].set_title('Predicted Yaw Rate')
            self.axs[2, 1].legend()
            self.axs[2, 1].grid(True)

            # =====================
            # (H) Ground-Truth Yaw Rate: [2, 2]
            # =====================
            # You can estimate yaw rate from gt_psi_plot by finite difference
            dt = 0.1  # assume 0.1s interval, adapt as needed
            for i in range(gt_psi_plot.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]
                # gt_psi_plot[i, :valid_len, 0] is the heading angle
                # approximate yaw rate:
                yaw_series = gt_psi_plot[i, :valid_len, 0]
                try:
                    yaw_rate = np.gradient(yaw_series, dt)
                except Exception as e:
                    print(f"Failed to compute yaw_rate: {e}")
                    yaw_rate = np.zeros_like(yaw_series)
                self.axs[2, 2].plot(yaw_rate, marker='o',
                                    label=f'Veh {i + 1}',
                                    color=color)
            self.axs[2, 2].set_xlabel('Time Step')
            self.axs[2, 2].set_ylabel('Yaw Rate [rad/s]')
            self.axs[2, 2].set_title('Ground Truth Yaw Rate')
            self.axs[2, 2].legend()
            self.axs[2, 2].grid(True)

            # We won't use [2,0], so you can set a title or leave it blank
            self.axs[2, 0].set_title("Extra Plot or Empty")

            # Tighten up
            self.fig.tight_layout()

            # Force matplotlib to update
            self.fig.canvas.draw_idle()

        # Define a click event handler
        def onclick(event):
            # If the user clicked on the figure, get data coords
            if event.xdata is None or event.ydata is None:
                return
            x_click = event.xdata
            y_click = event.ydata
            if abs(x_click) < 0.1 and abs(y_click) < 0.1:
                return
            # We need to translate back from figure (matplotlib) coords to original "world" coords
            # The image is displayed from 0..image_width in x, 0..image_height in y
            # We'll do an approximate approach here, because the display might have margins:
            # We know the axis for the image is from (0..W, 0..H).
            # Just treat (x_click, y_click) as pixel coordinates in the displayed image.
            # Then invert the transformation: new_goal_x = (x_click / resolution[0]) - center[0], etc.

            # The sample image is resolution: dataset_dict[0]['bbox_pixel']
            # We can do a direct scale:
            #   new_goal_x = x_click/resolution[0] - center[0]
            #   new_goal_y = ( (image_height - y_click)/resolution[1] ) - center[1]
            # Because we used int(...) in forward plotting. We'll attempt the inverse:

            W_img = dataset_dict[0]['bbox_pixel'][0]
            H_img = dataset_dict[0]['bbox_pixel'][1]
            center = dataset_dict[0]['center_meter']
            resolution = np.array(dataset_dict[0]['bbox_pixel']) / np.array(dataset_dict[0]['bbox_meter'])

            new_goal_x = (x_click / resolution[0]) - center[0]
            # For y, remember: Y was displayed top -> bottom in images, we did:
            #   y_pixel = int(H_img - (gy + center[1]) * resolution[1])
            # So invert:
            #   y_click = H_img - (world_y + center[1]) * resolution[1]
            #   => world_y = (H_img - y_click)/resolution[1] - center[1]

            new_goal_y = (H_img - y_click)/resolution[1] - center[1]

            print(f"Setting new goal for vehicle={self.current_vehicle_idx} at (x={new_goal_x:.2f}, y={new_goal_y:.2f})")

            # Re-run inference and re-draw
            run_inference_and_plot(new_goal_x, new_goal_y)

        # Initialize the figure and connect the event
        run_inference_and_plot()

        def on_key(event):
            if event.key == 'r':
                print("Key 'r' pressed — re-running inference.")
                run_inference_and_plot()
        # cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

        # ax_dropdown = inset_axes(self.axs[2, 0], width="100%", height="100%", loc="center")
        # vehicle_labels = [str(i) for i in range(1,max_vehicle_idx + 1)]  # Create labels
        # radio = RadioButtons(ax_dropdown, vehicle_labels)
        kid = self.fig.canvas.mpl_connect('key_press_event', on_key)
        # Connect the radio button event
        # radio.on_clicked(update_vehicle)
        # --- NEW SLIDER AXIS ---
        ax_slider = inset_axes(self.axs[2, 0],
                               width="80%", height="100%",
                               loc="lower left",
                               bbox_to_anchor=(0.1, 0.05, 0.8, 0.1),
                               bbox_transform=self.axs[2, 0].transAxes)

        temp_slider = Slider(
            ax=ax_slider,
            label="Temp",
            valmin=0.001,
            valmax=20.0,
            valinit=self.temperature,
            valstep=0.05
        )

        def update_temp(val):
            self.temperature = val
            print(f"Temperature set to {val:.2f}, re-running inference…")
            run_inference_and_plot()  # uses self.temperature internally

        temp_slider.on_changed(update_temp)

        # Show the interactive figure (blocking). When the user closes, we end.
        print("Interactive mode: Click on the figure to set new goal points. Close figure to end.")
        plt.show()

        # Once figure is closed, we're done.
        # self.fig.canvas.mpl_disconnect(cid)
        print("User closed figure. Evaluation complete.")

        return None  # or return something else if desired
