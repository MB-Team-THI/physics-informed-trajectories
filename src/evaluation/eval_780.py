import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.evaluation.eval_utils import compute_sampling_metrics

# Drop-in replacement for the older eval_730 in your code base
# We rename the same class "eval_740" and keep the method signatures.
# Comments are left largely intact, with the modifications highlighted.

UPSAMPLE_FACTOR = 16
NUM_ROLLOUTS = 32
SCENARIO_IDX = 5
VEHICLE_IDX = 2
TEMPERATURE = 10
SAMPLING_MODE = "multinomial"
MAX_SAMPLES = 50

SCENARIO_IDX = str(SCENARIO_IDX + 1)
VEHICLE_IDX = str(VEHICLE_IDX + 1)


class eval_780:
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
        self.metric_to_display_idx = 0  # 0: Spread, 1: Diversity, 2: Multimodality
        self.metric_names = ["Spread", "Diversity", "Multimodality"]

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
            if batch_idx > MAX_SAMPLES:
                break
        if not all_samples:
            print("No data in dataloader. Exiting.")
            return None

        # Ask user for scenario index
        max_scenario_idx = len(all_samples)
        scenario_idx_str = SCENARIO_IDX#input(f"Enter scenario index [1..{max_scenario_idx}] (default=1): ")
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
        vehicle_idx_str = VEHICLE_IDX #input("Enter single vehicle index [1..N] (default=1): ")
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
                    target_length=self.target_len,
                    temperature=TEMPERATURE,
                    sample_mode=SAMPLING_MODE
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
        # For convenience, mark the base run's goal for the chosen vehicle (whatever it was)
        gx0 = self.sample_cache['cond_goal_point'][frame_idx, vehicle_idx, 0].item()
        gy0 = self.sample_cache['cond_goal_point'][frame_idx, vehicle_idx, 1].item()
        for _ in range(NUM_ROLLOUTS):
            base_run = run_model_for_current_sample()
            base_run['goal_x'] = gx0
            base_run['goal_y'] = gy0
            self.all_predictions.append(base_run)
        self.initial_predictions = self.all_predictions.copy()

        # Now define a function to re-draw everything we have so far
        def plot_all_predictions():
            # We'll build a fresh figure or clear the old one
            if self.fig is None or self.axs is None:  # Corrected from self.axs
                self.fig, self.axs = plt.subplots(1, 1, figsize=(16, 12))
            else:
                self.axs.clear()  # Clear the single axis

            # --- Setup for plotting ---
            frame_idx = 0  # Assuming we are always plotting the first frame in the batch
            center = self.dataset_dict[0]['center_meter']
            resolution = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            H_img = self.dataset_dict[0]['bbox_pixel'][1]

            upsample_factor = UPSAMPLE_FACTOR
            resolution *= upsample_factor
            H_img = int(H_img * upsample_factor)

            # --- Prepare background image ---
            image = self.sample_cache['images'][frame_idx, 0, :, :].cpu().numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_rgb = cv2.resize(image_rgb, None, fx=upsample_factor, fy=upsample_factor,
                                   interpolation=cv2.INTER_LINEAR)

            # --- Plotting constants ---
            num_vehicles = self.sample_cache['hist_obj_lens'][frame_idx]
            obj_lengths_sum = self.sample_cache['hist_object_lengths_sum']
            traj_hist = self.sample_cache['hist_objs'][frame_idx].cpu().numpy()
            hist_lens = self.sample_cache['hist_objs_seq_len'][
                        obj_lengths_sum[frame_idx]:obj_lengths_sum[frame_idx + 1]]
            pred_lens = self.sample_cache['pred_objs_seq_len'][
                        obj_lengths_sum[frame_idx]:obj_lengths_sum[frame_idx + 1]]
            run_colors = list(plt.get_cmap('tab10').colors)
            hist_color = (0.6, 0.6, 0.6)  # Gray for history

            # --- Main Loop: Iterate over GROUPS of predictions (one group per goal) ---
            num_groups = (len(self.all_predictions) + NUM_ROLLOUTS - 1) // NUM_ROLLOUTS

            for group_idx in range(num_groups):
                start_idx = group_idx * NUM_ROLLOUTS
                end_idx = start_idx + NUM_ROLLOUTS
                runs_for_group = self.all_predictions[start_idx:end_idx]

                if not runs_for_group: continue

                # --- 1. Compute and Display Metrics for this Group ---
                metrics = compute_sampling_metrics(runs_for_group)
                color = run_colors[group_idx % len(run_colors)]

                if metrics is not None:
                    # Select the specific metric value for the active vehicle
                    metric_val = metrics[self.current_vehicle_idx][self.metric_to_display_idx]
                    metric_name = self.metric_names[self.metric_to_display_idx]
                    text = f"{metric_name[0]}: {metric_val:.2f}"  # Use first letter for brevity (e.g., S: 12.34)

                    # Get goal coordinates to position the text
                    goal_x = runs_for_group[0]['goal_x']
                    goal_y = runs_for_group[0]['goal_y']
                    gx_pix = int((goal_x + center[0]) * resolution[0])
                    gy_pix = int(H_img - (goal_y + center[1]) * resolution[1])

                    # Position text above the goal marker
                    text_pos = (gx_pix, gy_pix - 10 * upsample_factor)

                    # Draw a background rectangle for the text for better visibility
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3 * upsample_factor, 2)
                    # cv2.rectangle(image_rgb, (text_pos[0], text_pos[1] - h), (text_pos[0] + w, text_pos[1]), color, -1)

                    # Draw the text itself
                    # cv2.putText(image_rgb, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    #             fontScale=0.3 * upsample_factor, color=(1, 1, 1), thickness=2)  # White text

                    # Draw the goal marker
                    cv2.drawMarker(image_rgb, (gx_pix, gy_pix), color,
                                   markerType=cv2.MARKER_TILTED_CROSS, markerSize=8 * upsample_factor,
                                   thickness=2 * upsample_factor)

                # --- 2. Draw Trajectories for each run in this group ---
                for run_dict in runs_for_group:
                    X_plot = run_dict['X_reshaped'][frame_idx].cpu().numpy()
                    Y_plot = run_dict['Y_reshaped'][frame_idx].cpu().numpy()

                    # Plot trajectory ONLY for the currently active vehicle
                    veh_id = self.current_vehicle_idx
                    p_len = pred_lens[veh_id]
                    pred_x, pred_y = X_plot[veh_id, :p_len, 0], Y_plot[veh_id, :p_len, 0]

                    pred_coords = []
                    for (xx, yy) in zip(pred_x, pred_y):
                        px = int((xx + center[0]) * resolution[0])
                        py = int(H_img - (yy + center[1]) * resolution[1])
                        pred_coords.append((px, py))

                    if pred_coords:
                        pred_coords = np.array(pred_coords).reshape(-1, 1, 2)
                        image_rgb = cv2.polylines(image_rgb, [pred_coords.astype(np.int32)], isClosed=False,
                                                  color=color, thickness=2 * upsample_factor)

            # --- 3. Draw History (once for the active vehicle) ---
            veh_id = self.current_vehicle_idx
            h_len = hist_lens[veh_id]
            x_hist, y_hist = traj_hist[veh_id, :h_len, 0], traj_hist[veh_id, :h_len, 1]
            hist_coords = []
            for (xh, yh) in zip(x_hist, y_hist):
                px = int((xh + center[0]) * resolution[0])
                py = int(H_img - (yh + center[1]) * resolution[1])
                hist_coords.append((px, py))

            if hist_coords:
                hist_coords = np.array(hist_coords).reshape(-1, 1, 2)
                image_rgb = cv2.polylines(image_rgb, [hist_coords.astype(np.int32)], isClosed=False,
                                          color=hist_color, thickness=3 * upsample_factor)
            gt_x = self.sample_cache['pred_objsx'][frame_idx, veh_id, :, 0].cpu().numpy()
            gt_y = self.sample_cache['pred_objsy'][frame_idx, veh_id, :, 0].cpu().numpy()

            gt_coords = []
            for (gx, gy) in zip(gt_x, gt_y):
                if np.isnan(gx) or np.isnan(gy):
                    continue
                px = int((gx + center[0]) * resolution[0])
                py = int(H_img - (gy + center[1]) * resolution[1])
                gt_coords.append((px, py))

            gt_coords = np.array(gt_coords).reshape(-1, 1, 2)
            image_rgb = cv2.polylines(image_rgb, [gt_coords.astype(np.int32)], isClosed=False,
                                      color=(1.0, 0.0, 0.0), thickness=4)  # red line

            # --- 4. Final Display ---
            self.axs.imshow(image_rgb)
            # --- 3b. Draw Ground Truth trajectory for comparison ---


            self.axs.axis("off")
            self.axs.set_title(
                f"Scenario {self.current_scenario_idx + 1}, Vehicle {self.current_vehicle_idx + 1} | Metric: {self.metric_names[self.metric_to_display_idx]}")

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

        # Define a scroll event handler for zooming
        def on_scroll(event):
            # Only apply zoom if the scroll event occurs in the top-left subplot
            if event.inaxes != self.axs:
                return

            ax = self.axs
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
        def on_key(event):
            if event.key == 'r':
                print("Resetting to base scenario (key press: 'r')")
                # CORRECT: Reset to the stored initial set of trajectories
                self.all_predictions = self.initial_predictions.copy()
                plot_all_predictions()
            if event.key == 'm':
                # Cycle to the next metric index (0 -> 1 -> 2 -> 0)
                self.metric_to_display_idx = (self.metric_to_display_idx + 1) % len(self.metric_names)
                metric_name = self.metric_names[self.metric_to_display_idx]
                print(f"Switched to visualizing metric: {metric_name}")
                # Redraw the plot to show the new metric
                plot_all_predictions()
        def on_click(event):
            if event.inaxes != self.axs:
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
            for _ in range(NUM_ROLLOUTS):
                new_run = run_model_for_current_sample()
                new_run['goal_x'] = new_goal_x
                new_run['goal_y'] = new_goal_y
                self.all_predictions.append(new_run)

                # Re-plot everything
            plot_all_predictions()

            # Re-plot everything
            plot_all_predictions()

        # First plot (the base scenario) before any clicks:
        plot_all_predictions()

        # Connect the callback, wait for user to click multiple times
        cid = self.fig.canvas.mpl_connect('button_press_event', on_click)
        cid_scroll = self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', on_key)


        print("Interactive mode: click in the top-left subplot to set new goals. Close figure to end.")
        plt.show()  # block until user closes

        # Once closed, disconnect
        self.fig.canvas.mpl_disconnect(cid)
        print("User closed figure. Evaluation complete.")
        # import tikzplotlib
        # tikzplotlib.save("trajectory_plot.tex")
        return None
