import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, TextBox
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Drop-in replacement for the older eval_730 in your code base
# We rename the same class "eval_740" and keep the method signatures.
# Comments are left largely intact, with the modifications highlighted.

UPSAMPLE_FACTOR = 8
class eval_770:
    """
    This class extends the functionality of eval_730 by allowing a user to:
      - Choose a scenario (batch index) from the console
      - Preselect a single vehicle index from the console
      - Then interactively click on the displayed map up to N times (e.g. 4).
        Each click sets a new goal point *only* for that chosen vehicle.

    After each new goal point is set, the model re-runs on the fly, generating
    a fresh predicted trajectory for *all* vehicles (with the chosen vehicle’s
    goal updated). We accumulate each of those predicted trajectories and plot
    them (the single chosen vehicle's trajectory changes color each time; other
    vehicles also appear in that same color, so each "run" is a consistent color).

    Usage:
        e = eval_740(...)
        e(model=..., dataloader_test=..., device=..., dataset_dict=...)

    Once _evaluate(...) is called, a small interactive window with matplotlib
    will appear.
    The user will:
      1) Input a scenario index in the console
      2) Input a single vehicle index in the console
      3) A figure appears.  Click up to N times on the top-left subplot to set new goal points
      4) Each time you click, the model re-runs, the figure is updated with a new color for that run
      5) Close the figure window when done (loop ends).

    Note:
        This is a simplistic demonstration of interactive Matplotlib usage.  In practice,
        you could build a more robust UI using frameworks like Gradio, Plotly Dash, or Streamlit.
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
        self.idx = idx
        self.name = name
        self.input_ = input_
        self.output = output
        self.description = description

        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model

        # We'll store references after __call__ is invoked
        self.model = None
        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None

        # Scenario and vehicle indices
        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0

        # How many time steps to predict
        self.target_len = 30

        # Store figure & axes for repeated updates
        self.fig = None
        self.axs = None

        # We will store results of each run (the base scenario plus additional runs).
        # Each entry in self.all_predictions will be a dictionary with keys:
        #   {
        #       'X_reshaped':   shape [num_vehicles, pred_len, 1],
        #       'Y_reshaped':   shape [num_vehicles, pred_len, 1],
        #       'psi_dot':      shape [num_vehicles, pred_len],
        #       'ax':           shape [num_vehicles, pred_len],
        #       'v':            shape [num_vehicles, pred_len],
        #       'goal_x':       float (the newly clicked goal for the chosen vehicle),
        #       'goal_y':       float,
        #   }
        # We use one color per entry in that list.
        self.all_predictions = []

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        """
        Called externally.  We store references for interactive usage.
        """
        self.model = model
        self.dataloader_test = dataloader_test
        # Force to cuda if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_dict = dataset_dict

        return self._evaluate(
            self.model,
            self.dataloader_test,
            self.device,
            self.dataset_dict
        )

    @staticmethod
    def _process_dynamic_model_reshape(output, gTruthX, gTruthY, gTruthT,
                                       x_traj_pred_obj_len, pres_object_lengths_sum):
        # Same helper code as in eval_730
        X = output['X']
        Y = output['Y']
        T = output['T']

        X_reshaped = torch.zeros_like(gTruthX, device=gTruthX.device)
        Y_reshaped = torch.zeros_like(gTruthY, device=gTruthY.device)
        T_reshaped = torch.zeros_like(gTruthT, device=gTruthT.device)

        for unp in range(gTruthX.shape[0]):
            start = pres_object_lengths_sum[unp]
            end   = pres_object_lengths_sum[unp+1]
            this_len = x_traj_pred_obj_len[unp]

            X_reshaped[unp, 0:this_len, :, :] = X[start:end, :].unsqueeze(2)
            Y_reshaped[unp, 0:this_len, :, :] = Y[start:end, :].unsqueeze(2)
            T_reshaped[unp, 0:this_len, :, :] = T[start:end, :].unsqueeze(2)

        # We then overwrite with ground truth
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
        Y_reshaped = torch.empty_like(gTruthY, device=gTruthY.device)
        T_reshaped = torch.empty_like(gTruthT, device=gTruthT.device)

        for unp in range(gTruthX.shape[0]):
            start = pres_object_lengths_sum[unp]
            end   = pres_object_lengths_sum[unp+1]
            this_len = x_traj_pred_obj_len[unp]

            X_reshaped[unp, 0:this_len, :, :] = X[start:end, :].unsqueeze(2)
            Y_reshaped[unp, 0:this_len, :, :] = Y[start:end, :].unsqueeze(2)
            T_reshaped[unp, 0:this_len, :, :] = T[start:end, :].unsqueeze(2)

        return X_reshaped, Y_reshaped, T_reshaped

    def _evaluate(self, model, dataloader_test, device, dataset_dict, frame_idx=0):
        """
        Main evaluation function.
        1) Let user pick scenario from console.
        2) Let user pick a single vehicle from console.
        3) Show interactive figure. User can click multiple times to set new goal points
           for that single vehicle. Each time, we re-run the model, storing the entire
           scenario's predictions in a new color. We keep overlaying them in the figure.
        4) Close the figure to end.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        if next(model.parameters()).device != device:
            model.to(device)

        # Convert the dataloader to a small list of samples (just to pick from)
        all_samples = []
        for batch_idx, sample in enumerate(tqdm(dataloader_test(epoch=0))):
            all_samples.append(sample)
            if batch_idx > 20:
                break
        if not all_samples:
            print("No data in dataloader. Exiting.")
            return None

        # Ask user for scenario index
        max_scenario_idx = len(all_samples)
        scenario_idx_str = input(f"Enter scenario index [1..{max_scenario_idx}] (default=1): ")
        if scenario_idx_str.strip() == "":
            scenario_idx = 0
        else:
            scenario_idx = int(scenario_idx_str) - 1
        if scenario_idx < 0 or scenario_idx >= max_scenario_idx:
            scenario_idx = 0
        self.current_scenario_idx = scenario_idx

        # We only take that one scenario (batch).
        sample = all_samples[scenario_idx]
        self.sample_cache = sample

        # Ask user for single vehicle index
        # (You can clamp to the actual # vehicles, but let's assume there's at least 1)
        vehicle_idx_str = input("Enter single vehicle index [1..N] (default=1): ")
        if vehicle_idx_str.strip() == "":
            vehicle_idx = 0
        else:
            vehicle_idx = int(vehicle_idx_str) - 1
        if vehicle_idx < 0:
            vehicle_idx = 0
        self.current_vehicle_idx = vehicle_idx

        # We define a function that will run the model *once* with the current sample_cache
        # and store the entire scenario's predictions. We'll then accumulate them in self.all_predictions.
        def run_model_for_current_sample():
            # Prepare sample for inference
            x_image = self.sample_cache['images'].to(device)
            x_traj = [self.sample_cache['hist_objs'].to(device), self.sample_cache['hist_obj_lens']]
            x_traj_len = self.sample_cache['hist_objs_seq_len']
            x_traj_pred_obj_len = self.sample_cache['pred_obj_lens']
            x_traj_pred_len = self.sample_cache['pred_objs_seq_len']
            pres_object_lengths_sum = self.sample_cache['pres_object_lengths_sum']
            obj_length_padded = self.sample_cache['hist_object_lengths_sum']

            batch_wise_decoder_input = self.sample_cache['obj_decoder_in'].to(device)
            gTruthX = self.sample_cache['pred_objsx'].to(device)
            gTruthY = self.sample_cache['pred_objsy'].to(device)
            gTruthT = self.sample_cache['pred_objst'].to(device)

            # Additional ground-truth arrays (for velocity, acceleration, yaw)
            gt_v = self.sample_cache['pred_objsv'].to(device)
            gt_a_lat = self.sample_cache['pred_objs_a_lat'].to(device)
            gt_a_lon = self.sample_cache['pred_objs_a_lon'].to(device)
            gt_psi = self.sample_cache['pred_objspsi'].to(device)

            condition_goal_point = self.sample_cache['cond_goal_point'].to(device).float()
            condition_v = self.sample_cache['cond_v'].to(device).float()

            combined_condition = torch.cat([condition_goal_point], dim=2)

            with torch.no_grad():
                output = model(
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

            # We'll store the relevant outputs in a dictionary for plotting
            run_data = {
                'X_reshaped': X_reshaped,
                'Y_reshaped': Y_reshaped,
                'T_reshaped': T_reshaped,
                'psi_dot': output['psi_dot'],
                'ax': output['ax'],
                'v': output['v'],
            }
            return run_data

        # We'll do an initial "base" run with the scenario as-is:
        base_run = run_model_for_current_sample()
        # For convenience, mark the base run's goal for the chosen vehicle (whatever it was)
        gx0 = self.sample_cache['cond_goal_point'][frame_idx, vehicle_idx, 0].item()
        gy0 = self.sample_cache['cond_goal_point'][frame_idx, vehicle_idx, 1].item()
        base_run['goal_x'] = gx0
        base_run['goal_y'] = gy0
        self.all_predictions.append(base_run)

        # Now define a function to re-draw everything we have so far
        def plot_all_predictions():
            # We'll build a fresh figure or clear the old one
            if self.fig is None or self.axs is None:
                self.fig, self.axs = plt.subplots(3, 3, figsize=(16, 12))
            else:
                for row_ax in self.axs:
                    for a_item in row_ax:
                        a_item.clear()

            # Prepare to plot background (the grayscale image)
            idx_for_plot = frame_idx
            center = dataset_dict[0]['center_meter']
            resolution = np.array(dataset_dict[0]['bbox_pixel']) / np.array(dataset_dict[0]['bbox_meter'])
            H_img = dataset_dict[0]['bbox_pixel'][1]

            upsample_factor = UPSAMPLE_FACTOR  # Change this value as needed
            resolution = resolution * upsample_factor  # Adjust resolution accordingly
            H_img = int(H_img * upsample_factor)  # Update image height

            x_image = self.sample_cache['images'].to(device)
            image = x_image[idx_for_plot, 0, :, :].cpu().numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_rgb = cv2.resize(image_rgb, None, fx=upsample_factor, fy=upsample_factor,
                                   interpolation=cv2.INTER_LINEAR)

            # Number of runs we have so far (base + however many clicks)
            num_runs = len(self.all_predictions)

            # We define a color palette for each run.
            # If you want 4 distinct runs, you'll see 4 distinct colors, etc.
            # (Feel free to expand or customize.)
            # Each run uses exactly one color for ALL vehicles in that run,
            # so the chosen vehicle plus others share that run color.
            run_colors = list(plt.get_cmap('tab10').colors)

            # Also grab some needed lengths for plotting
            x_traj = [self.sample_cache['hist_objs'].to(device), self.sample_cache['hist_obj_lens']]
            x_traj_len = self.sample_cache['hist_objs_seq_len']
            x_traj_pred_len = self.sample_cache['pred_objs_seq_len']
            obj_length_padded = self.sample_cache['hist_object_lengths_sum']
            num_vehicles = x_traj[1][idx_for_plot]  # how many vehicles in this scenario

            # Gather historical positions
            traj_hist = x_traj[0][idx_for_plot].cpu().numpy()  # shape [num_vehicles, hist_len, 3?]
            x_traj_len_local = x_traj_len[obj_length_padded[idx_for_plot] : obj_length_padded[idx_for_plot+1]]
            x_traj_pred_len_local = x_traj_pred_len[obj_length_padded[idx_for_plot] : obj_length_padded[idx_for_plot+1]]

            # We'll accumulate arrays for "Position Error to GT" only if you like (comparing runs).
            # That said, the original code compared to ground truth.  We'll do so for each run
            # if desired. But to keep it simpler, let's just show the new predicted lines.
            # If you want to compare to GT, you can expand this again to do so.

            # We'll do a separate loop for each run:
            for run_idx, run_dict in enumerate(self.all_predictions):
                color = run_colors[run_idx % len(run_colors)]
                hist_color = (0.5, 0, 0.55)

                X_reshaped = run_dict['X_reshaped']
                Y_reshaped = run_dict['Y_reshaped']
                psi_dot    = run_dict['psi_dot']
                ax         = run_dict['ax']
                v          = run_dict['v']
                # Unpack them for the specific frame of interest
                X_reshaped_plot = X_reshaped[idx_for_plot].cpu().numpy()  # [num_vehicles, pred_len, 1]
                Y_reshaped_plot = Y_reshaped[idx_for_plot].cpu().numpy()  # [num_vehicles, pred_len, 1]
                psi_dot_plot    = psi_dot[obj_length_padded[idx_for_plot] : obj_length_padded[idx_for_plot+1], :].cpu().numpy()
                ax_plot         = ax[obj_length_padded[idx_for_plot] : obj_length_padded[idx_for_plot+1], :].cpu().numpy()
                v_plot          = v[obj_length_padded[idx_for_plot] : obj_length_padded[idx_for_plot+1], :].cpu().numpy()

                # Let's plot each vehicle's history + predicted trajectory in the top-left subplot
                # Overlaid on the background image.
                for veh_id in range(num_vehicles):
                    if veh_id != self.current_vehicle_idx:
                        continue
                    # Plot the history in 1-pixel thickness
                    h_len = x_traj_len_local[veh_id]
                    x_hist = traj_hist[veh_id, :h_len, 0]
                    y_hist = traj_hist[veh_id, :h_len, 1]

                    hist_coords = []
                    for (xh, yh) in zip(x_hist, y_hist):
                        px = int((xh + center[0]) * resolution[0])
                        py = int(H_img - (yh + center[1]) * resolution[1])
                        hist_coords.append((px, py))
                    hist_coords = np.array(hist_coords).reshape(-1, 1, 2)
                    image_rgb = cv2.polylines(
                        image_rgb, [hist_coords.astype(np.int32)],
                        isClosed=False,
                        color=hist_color,
                        thickness=1 * UPSAMPLE_FACTOR
                    )

                    # Now the predicted trajectory
                    p_len = x_traj_pred_len_local[veh_id]
                    pred_x = X_reshaped_plot[veh_id, :p_len, 0]
                    pred_y = Y_reshaped_plot[veh_id, :p_len, 0]

                    pred_coords = []
                    for (xx, yy) in zip(pred_x, pred_y):
                        px = int((xx + center[0]) * resolution[0])
                        py = int(H_img - (yy + center[1]) * resolution[1])
                        pred_coords.append((px, py))
                    pred_coords = np.array(pred_coords).reshape(-1, 1, 2)
                    image_rgb = cv2.polylines(
                        image_rgb, [pred_coords.astype(np.int32)],
                        isClosed=False,
                        color=color,
                        thickness=2 * UPSAMPLE_FACTOR
                    )

                # Also draw the newly used goal for the chosen vehicle in that color
                if run_idx == 0:
                    # This is the base scenario's original goal
                    gxx = run_dict['goal_x']
                    gyy = run_dict['goal_y']
                    if abs(gxx) > 1e-6 or abs(gyy) > 1e-6:
                        gx_pix = int((gxx + center[0]) * resolution[0])
                        gy_pix = int(H_img - (gyy + center[1]) * resolution[1])
                        cv2.drawMarker(
                            image_rgb, (gx_pix, gy_pix),
                            color,
                            markerType=cv2.MARKER_CROSS, markerSize=6 * UPSAMPLE_FACTOR, thickness=1 * UPSAMPLE_FACTOR
                        )
                else:
                    # For each additional run
                    gxx = run_dict['goal_x']
                    gyy = run_dict['goal_y']
                    gx_pix = int((gxx + center[0]) * resolution[0])
                    gy_pix = int(H_img - (gyy + center[1]) * resolution[1])
                    cv2.drawMarker(
                        image_rgb, (gx_pix, gy_pix),
                        color,
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=6 * UPSAMPLE_FACTOR, thickness=1 * UPSAMPLE_FACTOR
                    )

                # Next, fill the other subplots: velocity, accel, yaw rate, etc.
                # We'll do them in separate loops to avoid confusion. The code below
                # replicates the original structure, but uses the single color for each run.

                # (B) Predicted Velocity [0,1]
                for veh_id in range(num_vehicles):
                    if veh_id != self.current_vehicle_idx:
                        continue
                    valid_len = x_traj_pred_len_local[veh_id]
                    self.axs[0, 1].plot(
                        v_plot[veh_id, :valid_len],
                        marker='o', color=color,
                        label=f'Run{run_idx+1} Veh{veh_id+1}' if run_idx==0 and veh_id==0 else None
                    )

                # (E) Predicted Longitudinal Acceleration [1,1]
                for veh_id in range(num_vehicles):
                    if veh_id != self.current_vehicle_idx:
                        continue
                    valid_len = x_traj_pred_len_local[veh_id]
                    self.axs[1, 1].plot(
                        ax_plot[veh_id, :valid_len],
                        marker='o', color=color,
                        label=f'Run{run_idx+1} Veh{veh_id+1}' if run_idx==0 and veh_id==0 else None
                    )

                # (G) Predicted Yaw Rate [2,1]
                for veh_id in range(num_vehicles):
                    if veh_id != self.current_vehicle_idx:
                        continue
                    valid_len = x_traj_pred_len_local[veh_id]
                    self.axs[2, 1].plot(
                        psi_dot_plot[veh_id, :valid_len],
                        marker='o', color=color,
                        label=f'Run{run_idx+1} Veh{veh_id+1}' if run_idx==0 and veh_id==0 else None
                    )

            # Now that we've drawn all runs into image_rgb, show it at [0,0]
            self.axs[0, 0].imshow(image_rgb)
            self.axs[0, 0].axis("off")
            self.axs[0, 0].set_title(f"Trajectories (Scenario={self.current_scenario_idx}, Vehicle={self.current_vehicle_idx})")

            self.axs[0, 1].set_xlabel('Time Step')
            self.axs[0, 1].set_ylabel('Velocity [m/s]')
            self.axs[0, 1].set_title('Predicted Velocity')
            self.axs[0, 1].grid(True)

            self.axs[1, 1].set_xlabel('Time Step')
            self.axs[1, 1].set_ylabel('Accel [m/s^2]')
            self.axs[1, 1].set_title('Predicted Longitudinal Acceleration')
            self.axs[1, 1].grid(True)

            self.axs[2, 1].set_xlabel('Time Step')
            self.axs[2, 1].set_ylabel('Yaw Rate [rad/s]')
            self.axs[2, 1].set_title('Predicted Yaw Rate')
            self.axs[2, 1].grid(True)

            # We won't fill in [0,2], [1,0], [1,2], [2,0], [2,2] in detail for brevity
            self.axs[0, 2].set_title('Extra / GT Plots if needed')
            self.axs[1, 0].set_title('Extra / GT Plots if needed')
            self.axs[1, 2].set_title('Extra / GT Plots if needed')
            self.axs[2, 0].set_title('Extra / GT Plots if needed')
            self.axs[2, 2].set_title('Extra / GT Plots if needed')

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

        # Define a scroll event handler for zooming
        def on_scroll(event):
            # Only apply zoom if the scroll event occurs in the top-left subplot
            if event.inaxes != self.axs[0, 0]:
                return

            ax = self.axs[0, 0]
            # Get current limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Determine zoom direction: scroll up to zoom in, scroll down to zoom out.
            # Adjust zoom_factor as needed (e.g., 0.9 for zoom in, 1.1 for zoom out)
            if event.button == 'up':
                zoom_factor = 0.9  # zoom in
            elif event.button == 'down':
                zoom_factor = 1.1  # zoom out
            else:
                zoom_factor = 1.0

            # Use the mouse pointer as the center of the zoom
            if event.xdata is not None and event.ydata is not None:
                new_width = x_range * zoom_factor
                new_height = y_range * zoom_factor
                ax.set_xlim([event.xdata - new_width / 2, event.xdata + new_width / 2])
                ax.set_ylim([event.ydata - new_height / 2, event.ydata + new_height / 2])
                self.fig.canvas.draw_idle()

        # Connect the scroll event to the on_scroll handler

        # Define an event to handle clicks. Each click sets a new goal for the chosen vehicle.
        def on_click(event):
            if event.inaxes != self.axs[0, 0]:
                return  # only consider clicks in the top-left subplot area
            if event.xdata is None or event.ydata is None:
                return

            # Convert from figure coords to "world" coords
            W_img = dataset_dict[0]['bbox_pixel'][0]
            H_img = dataset_dict[0]['bbox_pixel'][1]
            center = dataset_dict[0]['center_meter']
            resolution = np.array(dataset_dict[0]['bbox_pixel']) / np.array(dataset_dict[0]['bbox_meter'])

            # x_pixel = event.xdata
            # y_pixel = event.ydata
            # In the original code: x_world = (x_pixel / resolution[0]) - center[0]
            upsample_factor = UPSAMPLE_FACTOR  # Change this value as needed
            resolution = resolution * upsample_factor  # Adjust resolution accordingly
            H_img = int(H_img * upsample_factor)
            new_goal_x = (event.xdata / resolution[0]) - center[0]
            # y_world = (H_img - y_pixel)/resolution[1] - center[1]
            new_goal_y = (H_img - event.ydata)/resolution[1] - center[1]

            print(f"New goal for vehicle={self.current_vehicle_idx}: (x={new_goal_x:.2f}, y={new_goal_y:.2f})")

            # Update the sample_cache so that the chosen vehicle has this new goal
            self.sample_cache['cond_goal_point'][frame_idx, self.current_vehicle_idx, 0] = new_goal_x
            self.sample_cache['cond_goal_point'][frame_idx, self.current_vehicle_idx, 1] = new_goal_y

            # Re-run the model and store the results
            new_run = run_model_for_current_sample()
            new_run['goal_x'] = new_goal_x
            new_run['goal_y'] = new_goal_y
            self.all_predictions.append(new_run)

            # Re-plot everything
            plot_all_predictions()

        # First plot (the base scenario) before any clicks:
        plot_all_predictions()

        # Connect the callback, wait for user to click multiple times
        cid = self.fig.canvas.mpl_connect('button_press_event', on_click)
        cid_scroll = self.fig.canvas.mpl_connect('scroll_event', on_scroll)


        print("Interactive mode: click in the top-left subplot to set new goals. Close figure to end.")
        plt.show()  # block until user closes

        # Once closed, disconnect
        self.fig.canvas.mpl_disconnect(cid)
        print("User closed figure. Evaluation complete.")
        # import tikzplotlib
        # tikzplotlib.save("trajectory_plot.tex")
        return None
