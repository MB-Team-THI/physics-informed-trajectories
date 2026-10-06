import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        p_all = {}
        batch_pass = 0
        max_id = 0
        center = dataset_dict[0]['center_meter']
        resolution = np.array(dataset_dict[0]['bbox_pixel']) / np.array(
            dataset_dict[0]['bbox_meter'])
        for batch_idx, sample in enumerate(dataloader_test(epoch=0)):
            # if batch_idx  > 99:
            #     continue
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
            x_offset = [0,0]
            y_offset = [0,0]
            vehicle_idx = [2,3]
            frame = frame_idx
            # for x_offset_i, y_offset_i, vehicle_idx_i in zip(x_offset, y_offset, vehicle_idx):
            #     sample["cond_goal_point"][frame][vehicle_idx_i][0] = sample["cond_goal_point"][frame][vehicle_idx_i][0] + x_offset_i
            #     sample["cond_goal_point"][frame][vehicle_idx_i][1] = sample["cond_goal_point"][frame][vehicle_idx_i][1] - y_offset_i
                # sample["cond_v"][frame][vehicle_idx_i][0] = sample["cond_v"][frame][vehicle_idx_i][0] + 15
                # print(sample["cond_v"][frame][vehicle_idx_i][0])


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
            gt_v, gt_a_lon, gt_a_lat = eval_730._process_dynamic_model_reshape(output,gt_v, gt_a_lon, gt_a_lat, x_traj_pred_obj_len, pres_object_lengths_sum)
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
                fig, axs = plt.subplots(3, 3, figsize=(16, 12))





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


                axs[0, 1].grid(True)

                x_traj_len_local = x_traj_len[obj_length_padded[idx]:obj_length_padded[idx + 1]]
                x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx]:obj_length_padded[idx + 1]]
                cmap_objec = get_cmap(X_reshaped_plot.shape[0])
                plot_v = []
                plot_a_lon = []
                plot_a_lat = []
                plot_psi_gt = []
                ade_diff = []
                for id, (x, y, x_temp_hist, y_temp_hist, gt_x, gt_y,gt_v_,gt_a_lat_, gt_a_lon_, gt_psi_) in \
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

                # Plot v with the same colors as the trajectories
                for i in range(v.shape[0]):
                    color = prediction_colors[i % len(prediction_colors)]
                    valid_len = x_traj_pred_len_local[i]
                    axs[0, 1].plot(v[i, :valid_len], marker='o', label=f'Vehicle {i + 1}', color=color)
                axs[0, 1].set_xlabel('Time Step')
                axs[0, 1].set_ylabel('Velocity [m/s]')
                axs[0, 1].set_title('Predicted Velocity')
                axs[0, 1].legend()

                # Plot psi_dots with the same colors as the trajectories
                for i in range(psi_dots.shape[0]):
                    color = prediction_colors[i % len(prediction_colors)]
                    valid_len = x_traj_pred_len_local[i]
                    axs[2, 1].plot(psi_dots[i, :valid_len], marker='o', label=f'Vehicle {i + 1}', color=color)
                axs[2, 1].set_xlabel('Time Step')
                axs[2, 1].set_ylabel('Yaw Rate [rad/s]')  # Psi dot
                axs[2, 1].set_title('Predicted Yaw Rate')
                axs[2, 1].legend()
                axs[2, 1].grid(True)

                # Plot ax with the same colors as the trajectories
                for i in range(ax.shape[0]):
                    color = prediction_colors[i % len(prediction_colors)]
                    valid_len = x_traj_pred_len_local[i]
                    axs[1, 1].plot(ax[i, :valid_len], marker='o', label=f'Vehicle {i + 1}', color=color)
                axs[1, 1].set_xlabel('Time Step')
                axs[1, 1].set_ylabel('Acceleration [m/s^2]')
                axs[1, 1].set_title('Predicted Longitudinal Acceleration')
                axs[1, 1].legend()
                axs[1, 1].grid(True)

                # Plot goal points
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
                    cv2.drawMarker(image, (x, y), (255, 0, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

                # Display the trajectory (image) plot on axs[0, 0]
                axs[0, 0].imshow(image)
                axs[0, 0].axis('off')  # Hide axes
                axs[0, 0].set_title('Trajectory Visualization')

                plot_a_lon = torch.tensor(plot_a_lon)
                plot_a_lat = torch.tensor(plot_a_lat)

                # Plot the difference to ground truth (in meters)
                for i in range(len(ade_diff)):
                    color = prediction_colors[i % len(prediction_colors)]
                    valid_len = x_traj_pred_len_local[i]
                    axs[1, 0].plot(ade_diff[i][:valid_len], marker='o', label=f'Vehicle {i + 1}', color=color)
                axs[1, 0].set_xlabel('Time Step')
                axs[1, 0].set_ylabel('Difference [m]')
                axs[1, 0].set_title('Position Difference to Ground Truth')
                axs[1, 0].legend()
                axs[1, 0].grid(True)

                # Plot velocity ground truth
                for i, tensor in enumerate(plot_v):
                    color = prediction_colors[i % len(prediction_colors)]
                    axs[0, 2].plot(range(tensor.shape[0]), tensor, marker='o',
                                   label=f'Vehicle {i + 1}', color=color)
                axs[0, 2].set_xlabel('Time Step')
                axs[0, 2].set_ylabel('Velocity [m/s]')
                axs[0, 2].set_title('Ground Truth Velocity')
                axs[0, 2].legend()
                axs[0, 2].grid(True)

                # Compute yaw rate from ground-truth Psi (using a 0.1s interval in np.gradient) and plot
                plot_psi_dot_gt = [np.gradient(tensor.squeeze(), 0.1) for tensor in plot_psi_gt]
                for i, tensor in enumerate(plot_psi_dot_gt):
                    color = prediction_colors[i % len(prediction_colors)]
                    axs[2, 2].plot(range(tensor.shape[0]), tensor, marker='o',
                                   label=f'Vehicle {i + 1}', color=color)
                axs[2, 2].set_xlabel('Time Step')
                axs[2, 2].set_ylabel('Yaw Rate [rad/s]')
                axs[2, 2].set_title('Ground Truth Yaw Rate')
                axs[2, 2].legend()
                axs[2, 2].grid(True)

                # Plot acceleration ground truth
                for i in range(plot_a_lon.shape[0]):
                    color = prediction_colors[i % len(prediction_colors)]
                    axs[1, 2].plot(plot_a_lon[i], marker='o', label=f'Vehicle {i + 1}', color=color)
                axs[1, 2].set_xlabel('Time Step')
                axs[1, 2].set_ylabel('Acceleration [m/s^2]')
                axs[1, 2].set_title('Ground Truth Longitudinal Acceleration')
                axs[1, 2].legend()
                axs[1, 2].grid(True)

                plt.tight_layout()
                return fig


                def on_key(event):
                    if event.key == 'q':
                        print("Pressed 'q'. Closing plot.")
                        plt.close(event.canvas.figure)
                print(F"Scenario Number: {batch_idx +1}")
                fig = plt.gcf()
                fig.canvas.mpl_connect('key_press_event', on_key)
                return fig
                # plt.show()

                image_to_display = image
