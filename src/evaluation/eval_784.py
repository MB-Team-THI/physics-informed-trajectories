import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from einops import repeat
from matplotlib.widgets import Slider,RadioButtons
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

UPSAMPLE_FACTOR = 8
NUM_SAMPLES = 64
NUM_ROLLOUTS = 640
assert NUM_ROLLOUTS % NUM_SAMPLES ==0
NUM_ROLLOUTS //= NUM_SAMPLES
import os
MODES = ["argmax", "topk", "topp", "multinomial"]  # whatever your model supports
def _rep(data):
    if NUM_SAMPLES == 1:
        return data
    if isinstance(data, torch.Tensor):
        data = repeat(data, 'b ... -> (n b) ... ', n=NUM_SAMPLES)
        return data
    elif isinstance(data, list):
        return data * NUM_SAMPLES
def _squeeze_traj_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Input shape expected: [B, N, 1, T] (as produced by _process_dynamic_model).
    Returns: [B, N, T]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,N,1,T]; got {x.shape}")
    return x.squeeze(2)
class eval_784:
    """
    Interactive, nondeterministic trajectory evaluator with live temperature control
    + uncertainty metrics aggregation over multiple rollouts for the selected vehicle.
    """

    def __init__(self,
                 idx=121,
                 name='Interactive nondet scenario evaluation',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=True,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 num_samples=NUM_ROLLOUTS,
                 init_temperature=1.0,
                 description='Click to set a nondeterministic goal and run multiple samples'):
        self.idx = idx
        self.name = name
        self.input_ = input_
        self.output = output
        self.description = description

        self.visualize = visualize
        self.onlyEgo = onlyEgo
        self.dynamic_model = dynamic_model
        self.num_samples = num_samples

        # sampling “softness”
        self.temperature = init_temperature
        self.sample_mode = "topp"

        # Will be set in __call__
        self.model = None
        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None

        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0
        self.target_len = 30

        # Matplotlib handles
        self.fig = None
        self.ax_main = None  # main image axis
        self.ax = None       # alias for on_click compatibility

        # Right-side metric axes
        self.ax_time = None          # per-time mean ± std (all six metrics)
        self.ax_hist_hnorm = None    # histogram (time-avg) Hnorm yaw/ax
        self.ax_hist_pmax = None     # histogram (time-avg) pmax yaw/ax
        self.ax_hist_margin = None   # histogram (time-avg) margin yaw/ax
        self.ax_table = None         # summary table (mean±std across T and runs)

        # Accumulated predictions for the current goal-click
        self.all_predictions = []

    @staticmethod
    def _process_dynamic_model(output, gTruthX, gTruthY, gTruthT,
                               x_traj_pred_obj_len, pres_object_lengths_sum):
        X, Y, T = output['X'], output['Y'], output['T']
        Xr = torch.empty_like(gTruthX)
        Yr = torch.empty_like(gTruthY)
        Tr = torch.empty_like(gTruthT)
        for u in range(gTruthX.shape[0]):
            s = pres_object_lengths_sum[u]
            e = pres_object_lengths_sum[u+1]
            L = x_traj_pred_obj_len[u]
            Xr[u, :L,:,:] = X[s:e, :].unsqueeze(2)
            Yr[u, :L,:,:] = Y[s:e, :].unsqueeze(2)
            Tr[u, :L,:,:] = T[s:e, :].unsqueeze(2)
        return Xr, Yr, Tr

    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        self.model = model
        self.dataloader_test = dataloader_test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_dict = dataset_dict
        return self._evaluate()

    def _evaluate(self):
        # Prepare model & data
        self.model.eval()
        self.model.to(self.device)
        all_samples = []
        for i, sample in enumerate(tqdm(self.dataloader_test(epoch=0))):
            all_samples.append(sample)
            if i >= 50:
                break
        if not all_samples:
            print("No data in dataloader. Exiting.")
            return

        # Scenario selection
        max_idx = len(all_samples)
        s = "5"
        self.current_scenario_idx = max(0, min(max_idx-1, int(s)-1)) if s else 0
        self.sample_cache = all_samples[self.current_scenario_idx]

        # Vehicle selection
        v = "4"
        self.current_vehicle_idx = max(0, int(v)-1) if v else 0

        # Helper to run one forward pass
        def run_once():
            sc = self.sample_cache
            x_image = sc['images'].to(self.device)
            cond = torch.cat([sc['cond_goal_point'].to(self.device).float()], dim=2)
            x_traj_len = sc['hist_objs_seq_len']
            obj_length_padded = sc['hist_object_lengths_sum']  # prefix sums for original (B=1) sample
            batch_wise_decoder_input = sc['obj_decoder_in'].to(self.device)
            x_traj = [sc['hist_objs'].to(self.device), sc['hist_obj_lens']]

            # --- repeat inputs along batch for batched sampling ---
            x_image = _rep(x_image)  # (NUM_SAMPLES, C, H, W)
            x_traj = [_rep(x_traj[0]), _rep(x_traj[1])]  # hist_objs and its lens repeated
            x_traj_len = _rep(x_traj_len)  # seq lens per obj
            conditions = _rep(cond)
            batch_wise_decoder_input = _rep(batch_wise_decoder_input)

            # build new prefix sums for concatenated vehicle lists across replicas
            # Ns = number of vehicles in original sample 0
            Ns = int(obj_length_padded[1].item() - obj_length_padded[0].item())
            batch_wise_object_lengths_sum = torch.arange(NUM_SAMPLES + 1, device=self.device, dtype=torch.long) * Ns

            with torch.no_grad():
                out = self.model(
                    x_image=x_image,
                    x_traj=x_traj,
                    x_traj_len=x_traj_len,
                    batch_wise_object_lengths_sum=batch_wise_object_lengths_sum,
                    conditions=conditions,
                    batch_wise_decoder_input=batch_wise_decoder_input,
                    target_length=self.target_len,
                    temperature=self.temperature,
                    sample_mode=self.sample_mode
                )

            # reshape predictions back to (B=NUM_SAMPLES, Nmax, T, 1)
            Xr, Yr, Tr = self._process_dynamic_model(
                out,
                _rep(sc['pred_objsx'].to(self.device)),
                _rep(sc['pred_objsy'].to(self.device)),  # <-- was bug: use Y here
                _rep(sc['pred_objst'].to(self.device)),
                sc['pred_obj_lens'] * NUM_SAMPLES,  # repeat per-sample lens
                batch_wise_object_lengths_sum
            )

            # ----- Slice uncertainty metrics for the selected vehicle, per replica -----
            runs = []
            unc = out.get('uncertainty_metrics', None)
            per_tn = unc['per_tn'] if (unc is not None and 'per_tn' in unc) else None

            # original scenario offsets (for B=1) to know Ns; for replicas, we use blocks of size Ns
            veh = int(np.clip(self.current_vehicle_idx, 0, max(0, Ns - 1)))

            def sl_block(name, k):
                # per_tn[name]: [T, NUM_SAMPLES * Ns]; take kth block [k*Ns:(k+1)*Ns] and pick vehicle index
                arr = per_tn[name][:, k * Ns:(k + 1) * Ns][:, veh]  # [T]
                return arr.detach().cpu().numpy()

            for k in range(NUM_SAMPLES):
                # package each replica as an individual "run" dict, like before
                if per_tn is not None:
                    try:
                        unc_sel = {
                            'Hnorm_yaw': sl_block('Hnorm_yaw', k),
                            'Hnorm_ax': sl_block('Hnorm_ax', k),
                            'pmax_yaw': sl_block('pmax_yaw', k),
                            'pmax_ax': sl_block('pmax_ax', k),
                            'margin_yaw': sl_block('margin_yaw', k),
                            'margin_ax': sl_block('margin_ax', k),
                        }
                    except Exception as e:
                        print(f"[warn] Uncertainty slicing failed (replica {k}): {e}")
                        unc_sel = None
                else:
                    unc_sel = None

                # NOTE: Xr, Yr have shape [NUM_SAMPLES, Nmax, T, 1]; grab the kth slice
                runs.append({
                    'X_reshaped': Xr[k:k + 1],  # keep a batch dim for downstream code
                    'Y_reshaped': Yr[k:k + 1],
                    'psi_dot': out['psi_dot'][k * Ns:(k + 1) * Ns],  # per-vehicle per-step for this replica
                    'ax': out['ax'][k * Ns:(k + 1) * Ns],
                    'v': out['v'][k * Ns:(k + 1) * Ns],
                    'uncertainty_sel': unc_sel
                })

            return runs

        # ---- Figure & axes setup (grid) -----------------------------------------------
        def ensure_axes():
            if self.fig is not None:
                return
            self.fig = plt.figure(figsize=(18, 12))
            gs = self.fig.add_gridspec(
                3, 4,
                width_ratios=[2.2, 1, 1, 1],
                height_ratios=[1.2, 1, 1],
                wspace=0.35, hspace=0.35
            )
            # Main image occupies full height of first column
            self.ax_main = self.fig.add_subplot(gs[:, 0])
            self.ax = self.ax_main  # alias for event handler compatibility

            # Top row right: per-time mean±std (six metrics in one axis)
            self.ax_time = self.fig.add_subplot(gs[0, 1:])

            # Middle row: histograms of time-avg across rollouts
            self.ax_hist_hnorm  = self.fig.add_subplot(gs[1, 1])
            self.ax_hist_pmax   = self.fig.add_subplot(gs[1, 2])
            self.ax_hist_margin = self.fig.add_subplot(gs[1, 3])

            # Bottom row: summary table
            self.ax_table = self.fig.add_subplot(gs[2, 1:])
            self.ax_table.axis('off')

        # Plotting routine
        def plot_all():
            ensure_axes()

            # ==== Left main image (keep as is) ====
            img = self._get_background_image()
            self.ax_main.clear()
            self._draw_history(img, color=(.6, .6, .6))
            self.ax_main.imshow(img)
            self.ax_main.axis('off')

            all_predictions_tensor_x = [el["X_reshaped"] for el in self.all_predictions]
            all_predictions_tensor_y = [el["Y_reshaped"] for el in self.all_predictions]
            if len(all_predictions_tensor_x) > 0:
                all_predictions_tensor_x = torch.cat(all_predictions_tensor_x, dim=0)
                all_predictions_tensor_y = torch.cat(all_predictions_tensor_y, dim=0)
                all_predictions = torch.cat([all_predictions_tensor_x, all_predictions_tensor_y], dim=-1)
                trajs = all_predictions[:, self.current_vehicle_idx, ...]  # N, 30, 2
                trajs_np = trajs.cpu().numpy().reshape(trajs.shape[0], -1)
                kmeans = KMeans(n_clusters=3, random_state=0).fit(trajs_np)
                labels = kmeans.labels_  # array of length N

                # plot, coloring by rollout index (kept as in your code)
                cmap = plt.get_cmap('tab10').colors
                for i, run in enumerate(self.all_predictions):
                    color = cmap[i % len(cmap)]
                    self._draw_run(img, run, color=color, goal_point_color=(1.0, 0.1, 1.0))
                self.ax_main.imshow(img)

            # ==== Right-side uncertainty aggregation for selected vehicle ====
            # Collect per-time arrays across rollouts (shape [R, T])
            names = ['Hnorm_yaw', 'Hnorm_ax', 'pmax_yaw', 'pmax_ax', 'margin_yaw', 'margin_ax']
            stacks = {}
            R = 0
            for nm in names:
                vals = [r['uncertainty_sel'][nm] for r in self.all_predictions
                        if (r.get('uncertainty_sel') is not None and r['uncertainty_sel'].get(nm) is not None)]
                if len(vals) > 0:
                    stacks[nm] = np.stack(vals, axis=0)  # [R, T]
                    R = stacks[nm].shape[0]
                else:
                    stacks[nm] = None

            # Per-time mean ± std plot
            metrics_to_plot = ["Hnorm_yaw", "Hnorm_ax"]

            self.ax_time.clear()
            t_axis = None
            for nm in metrics_to_plot:
                arr = stacks.get(nm)
                if arr is None:
                    continue
                if t_axis is None:
                    t_axis = np.arange(arr.shape[1])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                self.ax_time.plot(t_axis, mean, label=nm)
                self.ax_time.fill_between(t_axis, mean - std, mean + std, alpha=0.2)

            self.ax_time.set_title(
                f'Normalized entropy over time — vehicle #{self.current_vehicle_idx + 1}'
            )
            self.ax_time.set_xlabel('Time step')
            self.ax_time.set_ylabel('Hnorm value')
            self.ax_time.legend()
            self.ax_time.grid(True)

            # Histograms of time-avg per rollout
            def timeavg(arr):
                return None if arr is None else arr.mean(axis=1)  # [R]

            self.ax_hist_hnorm.clear()
            hy = timeavg(stacks['Hnorm_yaw'])
            ha = timeavg(stacks['Hnorm_ax'])
            if hy is not None:
                self.ax_hist_hnorm.hist(hy, bins=12, alpha=0.6, label='Hnorm_yaw')
            if ha is not None:
                self.ax_hist_hnorm.hist(ha, bins=12, alpha=0.6, label='Hnorm_ax')
            self.ax_hist_hnorm.set_title('Time-avg Hnorm (per rollout)')
            self.ax_hist_hnorm.set_xlabel('Value'); self.ax_hist_hnorm.set_ylabel('Count')
            if hy is not None or ha is not None:
                self.ax_hist_hnorm.legend()
            self.ax_hist_hnorm.grid(True)

            self.ax_hist_pmax.clear()
            py = timeavg(stacks['pmax_yaw'])
            pa = timeavg(stacks['pmax_ax'])
            if py is not None:
                self.ax_hist_pmax.hist(py, bins=12, alpha=0.6, label='pmax_yaw')
            if pa is not None:
                self.ax_hist_pmax.hist(pa, bins=12, alpha=0.6, label='pmax_ax')
            self.ax_hist_pmax.set_title('Time-avg pmax (per rollout)')
            self.ax_hist_pmax.set_xlabel('Value'); self.ax_hist_pmax.set_ylabel('Count')
            if py is not None or pa is not None:
                self.ax_hist_pmax.legend()
            self.ax_hist_pmax.grid(True)

            self.ax_hist_margin.clear()
            my = timeavg(stacks['margin_yaw'])
            ma = timeavg(stacks['margin_ax'])
            if my is not None:
                self.ax_hist_margin.hist(my, bins=12, alpha=0.6, label='margin_yaw')
            if ma is not None:
                self.ax_hist_margin.hist(ma, bins=12, alpha=0.6, label='margin_ax')
            self.ax_hist_margin.set_title('Time-avg margin (per rollout)')
            self.ax_hist_margin.set_xlabel('Value'); self.ax_hist_margin.set_ylabel('Count')
            if my is not None or ma is not None:
                self.ax_hist_margin.legend()
            self.ax_hist_margin.grid(True)

            # Summary table: overall mean ± std across time and rollouts
            self.ax_table.clear(); self.ax_table.axis('off')
            rows = []
            for nm in names:
                arr = stacks[nm]
                if arr is None:
                    rows.append([nm, "—"])
                else:
                    m = arr.mean()
                    s = arr.std()
                    rows.append([nm, f"{m:.3f} ± {s:.3f}"])
            col_labels = ["Metric", "Mean ± Std (over time & rollouts)"]
            table = self.ax_table.table(cellText=rows, colLabels=col_labels, loc='center')
            table.scale(1, 1.3)
            self.ax_table.set_title('Uncertainty summary (selected vehicle)')

            self.fig.canvas.draw_idle()

        # Helpers to get image & draw
        def get_bg():
            sc = self.sample_cache
            idx = 0
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            H = self.dataset_dict[0]['bbox_pixel'][1]
            res *= UPSAMPLE_FACTOR; H *= UPSAMPLE_FACTOR
            img = sc['images'][idx,0].cpu().numpy()
            img = np.where(img==0, .2, img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return cv2.resize(img, None, fx=UPSAMPLE_FACTOR, fy=UPSAMPLE_FACTOR,
                              interpolation=cv2.INTER_LINEAR)
        self._get_background_image = get_bg

        def draw_hist(img, color):
            sc = self.sample_cache
            idx = 0
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            res *= UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR

            traj = sc['hist_objs'][idx].cpu().numpy()
            lengths = sc['hist_objs_seq_len'][sc['hist_object_lengths_sum'][idx]:sc['hist_object_lengths_sum'][idx+1]]
            for veh in [self.current_vehicle_idx]:
                try:
                    L = lengths[veh]
                except IndexError:
                    print(f"Vehicle {veh} has no history in this scenario.")
                    continue
                pts = traj[veh,:L,:2]
                pix = ((pts+center)*res).astype(int)
                pix[:,1] = H-pix[:,1]
                cv2.polylines(img, [pix.reshape(-1,1,2)], False, color,
                              thickness=2*UPSAMPLE_FACTOR)
        self._draw_history = draw_hist

        def draw_run(img, run, color, goal_point_color=None):
            sc = self.sample_cache
            idx = 0
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            res *= UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR

            Xp = run['X_reshaped'][idx].cpu().numpy()[:,:,0]
            Yp = run['Y_reshaped'][idx].cpu().numpy()[:,:,0]
            lengths = sc['pred_objs_seq_len'][sc['pres_object_lengths_sum'][idx]:sc['pres_object_lengths_sum'][idx+1]]
            veh = self.current_vehicle_idx
            try:
                L = lengths[veh]
            except IndexError:
                return
            pts = np.stack([Xp[veh,:L], Yp[veh,:L]], axis=-1)
            pix = ((pts+center)*res).astype(int)
            pix[:,1] = H-pix[:,1]
            cv2.polylines(img, [pix.reshape(-1,1,2)], False, color, thickness=4)

            gx, gy = sc['cond_goal_point'][0,veh]
            gp = np.array([gx,gy])
            pixg = ((gp+center)*res).astype(int)
            pixg[1] = H-pixg[1]
            c_goal = color if goal_point_color is None else goal_point_color
            cv2.drawMarker(img, tuple(pixg), c_goal,
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=8*UPSAMPLE_FACTOR,
                           thickness=1*UPSAMPLE_FACTOR)
        self._draw_run = draw_run

        # Mouse click handler (kept)
        def on_click(event):
            if event.inaxes != self.ax_main:
                return
            Wp, Hp = self.dataset_dict[0]['bbox_pixel']
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            res *= UPSAMPLE_FACTOR; Hp *= UPSAMPLE_FACTOR

            xw = event.xdata / res[0] - center[0]
            yw = (Hp - event.ydata) / res[1] - center[1]
            print(f"Setting new goal: ({xw:.2f}, {yw:.2f})")

            self.all_predictions.clear()
            self.sample_cache['cond_goal_point'][0, self.current_vehicle_idx, 0] = xw
            self.sample_cache['cond_goal_point'][0, self.current_vehicle_idx, 1] = yw
            for _ in trange(self.num_samples):
                self.all_predictions.extend(run_once())
            plot_all()

        # Key handler to reset (kept)
        def on_key(event):
            if event.key == 'r':
                print("Resetting predictions (press click again to sample).")
                self.all_predictions.clear()
                plot_all()

        # Initial draw
        plot_all()

        # Create UI widgets (kept; adjust bottom to leave space)
        plt.subplots_adjust(bottom=0.20)
        ax_temp = plt.axes([0.25, 0.05, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        temp_slider = Slider(
            ax=ax_temp, label='Temperature',
            valmin=0.0, valmax=5.0, valinit=self.temperature, valstep=0.5,
        )

        ax_scenario = plt.axes([0.10, 0.01, 0.35, 0.03], facecolor='lightgoldenrodyellow')
        scenario_slider = Slider(
            ax=ax_scenario, label='Scenario',
            valmin=1, valmax=len(all_samples),
            valinit=self.current_scenario_idx + 1, valstep=1,
        )

        ax_vehicle = plt.axes([0.55, 0.01, 0.35, 0.03], facecolor='lightgoldenrodyellow')
        vehicle_slider = Slider(
            ax=ax_vehicle, label='Vehicle',
            valmin=1, valmax=10,  # you can adapt to scenario-max dynamically
            valinit=self.current_vehicle_idx + 1, valstep=1,
        )

        def on_temp_change(val):
            self.temperature = val
            print(f"Temperature set to {val:.2f}. Re-sampling…")
            self.all_predictions.clear()
            for _ in trange(self.num_samples):
                self.all_predictions.extend(run_once())
            plot_all()
        temp_slider.on_changed(on_temp_change)

        def on_scenario_change(val):
            self.current_scenario_idx = int(val) - 1
            self.sample_cache = all_samples[self.current_scenario_idx]
            print(f"Switched to scenario {self.current_scenario_idx + 1}")
            self.all_predictions.clear()
            plot_all()
        scenario_slider.on_changed(on_scenario_change)

        def on_vehicle_change(val):
            self.current_vehicle_idx = int(val) - 1
            print(f"Switched to vehicle {self.current_vehicle_idx + 1}")
            self.all_predictions.clear()
            plot_all()
        vehicle_slider.on_changed(on_vehicle_change)

        # RadioButtons for sampling mode (kept)
        ax_mode = plt.axes([0.80, 0.05, 0.15, 0.15], facecolor='lightgoldenrodyellow')
        mode_radio = RadioButtons(ax_mode, MODES, active=MODES.index(self.sample_mode))

        def on_mode_change(label):
            self.sample_mode = label
            print(f"Sample mode set to {label}. Re-sampling…")
            self.all_predictions.clear()
            for _ in trange(self.num_samples):
                self.all_predictions.extend(run_once())
            plot_all()
        mode_radio.on_clicked(on_mode_change)

        # Connect events
        cid_click = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        cid_key   = plt.gcf().canvas.mpl_connect('key_press_event',   on_key)

        print("Click to sample trajectories; adjust sliders; press 'r' to clear; close window to exit.")
        plt.show()

        # cleanup
        plt.gcf().canvas.mpl_disconnect(cid_click)
        plt.gcf().canvas.mpl_disconnect(cid_key)
        print("Done.")
        return

