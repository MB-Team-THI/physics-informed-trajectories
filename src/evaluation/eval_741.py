from matplotlib.widgets import RadioButtons, Slider
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from src.evaluation.eval import eval
from tqdm import tqdm

class eval_741(eval):
    """
    Interactive scenario evaluation:
    - select a vehicle
    - modify its velocity with a slider
    - re-run the model after each adjustment
    """

    def __init__(self,
                 idx=121,
                 name='Interactive scenario evaluation with velocity control',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=True,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 description='Interactive scenario: set new velocity for a vehicle.'):
        super().__init__(idx, name, input_, output, description)
        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model

        self.model = None
        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None
        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0
        self.target_len = 30
        self.fig = None
        self.axs = None

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        self.model = model
        self.dataloader_test = dataloader_test
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_dict = dataset_dict
        return self._evaluate()

    @staticmethod
    def _process_dynamic_model(output, gTruthX, gTruthY, gTruthT, x_traj_pred_obj_len, pres_object_lengths_sum):
        X = output['X']
        Y = output['Y']
        T = output['T']

        X_reshaped = torch.empty_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.empty_like(gTruthY, device=gTruthY.device)
        T_reshaped = torch.empty_like(gTruthT, device=gTruthT.device)

        for unp in range(gTruthX.shape[0]):
            X_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = X[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1], :
            ].unsqueeze(2)
            Y_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = Y[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1], :
            ].unsqueeze(2)
            T_reshaped[unp, 0:x_traj_pred_obj_len[unp], :, :] = T[
                pres_object_lengths_sum[unp]:pres_object_lengths_sum[unp+1], :
            ].unsqueeze(2)

        return X_reshaped, Y_reshaped, T_reshaped

    def _evaluate(self, frame_idx=0):
        self.model.eval()
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # Collect a few batches
        all_samples = []
        for batch_idx, sample in enumerate(tqdm(self.dataloader_test(epoch=0))):
            all_samples.append(sample)
            if batch_idx >= 50:
                break
        if not all_samples:
            print("No data found. Exiting.")
            return

        # Pick scenario
        max_scenario_idx = len(all_samples)
        scenario_idx = input(f"Enter scenario index [1..{max_scenario_idx}] (default=1): ")
        scenario_idx = int(scenario_idx) - 1 if scenario_idx.strip() else 0
        scenario_idx = max(0, min(scenario_idx, max_scenario_idx - 1))
        self.current_scenario_idx = scenario_idx
        self.sample_cache = all_samples[scenario_idx]

        # Plotting helpers
        def run_inference_and_plot(new_v=None):
            # Prepare sample for inference
            if new_v is not None:
                self.sample_cache['cond_v'][0, self.current_vehicle_idx, 0] = float(new_v)
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
            combined_condition = torch.cat([condition_v], dim=2)

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

            # Also reshape ground truth arrays for velocity/accel/yaw so we can index them consistently
            # We'll use the same "3-tensor" reshape helper on each triple as needed.
            # If you need them individually, define a separate method or do them one by one.
            gt_v_reshaped, gt_a_lon_reshaped, gt_a_lat_reshaped = self._process_dynamic_model(
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
            gt_psi_reshaped, _, _ = eval_741._process_dynamic_model(
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
            gt_v_plot = gt_v_reshaped[idx][obj_length_padded[idx]: obj_length_padded[
                idx + 1]].cpu().numpy()  # [num_vehicles, pred_length, 1]
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
            cond_v_plot = self.sample_cache['cond_v'][idx].cpu().numpy()
            for i in range(v.shape[0]):
                color = prediction_colors[i % len(prediction_colors)]
                valid_len = x_traj_pred_len_local[i]

                # Average predicted v over prediction horizon
                mean_pred_v = v[i, :valid_len].mean()

                # Plot conditioned v (dashed line)
                self.axs[1, 0].hlines(
                    cond_v_plot[i, 0],
                    xmin=0, xmax=valid_len - 1,
                    linestyles='--',
                    colors=[color],
                    label=f'Cond Veh {i + 1}'
                )

                # Plot predicted mean v (solid line)
                self.axs[1, 0].hlines(
                    mean_pred_v,
                    xmin=0, xmax=valid_len - 1,
                    linestyles='-',
                    colors=[color],
                    label=f'Pred Veh {i + 1}'
                )
            self.axs[1, 0].set_xlabel('Time Step')
            self.axs[1, 0].set_ylabel('Velocity [m/s]')
            self.axs[1, 0].set_title('Conditioned vs Predicted Velocity')
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

        # Callbacks
        def update_vehicle(label):
            self.current_vehicle_idx = int(label) - 1
            slider.eventson = False
            slider.set_val(float(self.sample_cache['cond_v'][0, self.current_vehicle_idx, 0]))
            slider.eventson = True
            run_inference_and_plot()

        def update_velocity(val):
            run_inference_and_plot(new_v=val)

        # Initial plot
        run_inference_and_plot()

        # Add RadioButtons
        ax_dropdown = inset_axes(self.axs[2, 0], width="40%", height="100%", loc="center left")
        vehicle_labels = [str(i) for i in range(1, 16)]
        radio = RadioButtons(ax_dropdown, vehicle_labels)
        radio.on_clicked(update_vehicle)

        # Add Slider
        ax_slider = inset_axes(self.axs[2, 0], width="55%", height="30%", loc="center right")
        v_min, v_max = 0.0, 30.0
        init_v = float(self.sample_cache['cond_v'][0, self.current_vehicle_idx, 0])
        slider = Slider(ax_slider, 'v [m/s]', v_min, v_max, valinit=init_v)
        slider.on_changed(update_velocity)

        print("Interactive mode: adjust vehicle velocity with slider. Close window to finish.")
        plt.show()

        return None
