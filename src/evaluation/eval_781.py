import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from einops import repeat
from matplotlib.widgets import Slider,RadioButtons
from sklearn.cluster import KMeans
from tqdm import tqdm, trange

UPSAMPLE_FACTOR = 8
NUM_SAMPLES = 32
NUM_ROLLOUTS = NUM_SAMPLES * 1
assert NUM_ROLLOUTS % NUM_SAMPLES ==0
NUM_ROLLOUTS //= NUM_SAMPLES

PLOT_UNCERTAINTY = False

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

MODES = ["argmax", "topk", "topp", "multinomial"]  # whatever your model supports

class eval_781:
    """
    Interactive, nondeterministic trajectory evaluator with live temperature control.

    Usage:
        e = eval_781(num_samples=10, init_temperature=1.0)
        e(model=..., dataloader_test=..., device=..., dataset_dict=...)
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

        # new: sampling “softness”
        self.temperature = init_temperature
        self.sample_mode = "topp"

        # Will be set in __call__
        self.model = None
        self._uncert_cbar = None  # colorbar handle

        self.dataloader_test = None
        self.device = None
        self.dataset_dict = None
        self.sample_cache = None

        self.current_scenario_idx = 0
        self.current_vehicle_idx = 0
        self.target_len = 30

        # Matplotlib handles
        self.fig = None
        self.ax = None

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
                    'uncertainty_metrics': unc_sel
                })

            return runs

        # Plotting routine
        def plot_all():
            if self.fig is None:
                self.fig, self.ax = plt.subplots(1,1,figsize=(16,12))
            img = self._get_background_image()
            self.ax.clear()
            self._draw_history(img, color=(.6,.6,.6))


            self.ax.imshow(img)
            self.ax.axis('off')
            all_predictions_tensor_x = [el["X_reshaped"] for el in self.all_predictions]
            all_predictions_tensor_y = [el["Y_reshaped"] for el in self.all_predictions]
            all_predictions_uncertainties_yaw = [el["uncertainty_metrics"]['Hnorm_yaw'] for el in self.all_predictions]
            all_predictions_uncertainties_ax = [el["uncertainty_metrics"]['Hnorm_ax'] for el in self.all_predictions]

            #filter right uncert for vehicle
            all_predictions_uncertainties_yaw= [el[self.current_vehicle_idx].item() for el in all_predictions_uncertainties_yaw]
            all_predictions_uncertainties_ax= [el[self.current_vehicle_idx].item() for el in all_predictions_uncertainties_ax] # list of float
            from matplotlib import cm, colors as mcolors

            if len(all_predictions_tensor_x) > 0:
                all_predictions_tensor_x = torch.cat(all_predictions_tensor_x, dim=0)
                all_predictions_tensor_y = torch.cat(all_predictions_tensor_y, dim=0)
                all_predictions = torch.cat([all_predictions_tensor_x, all_predictions_tensor_y], dim=-1)
                trajs = all_predictions[:, self.current_vehicle_idx, ...]  # [N, 30, 2]

                # Build one scalar uncertainty per run (adjust aggregation if desired)
                U_yaw = np.asarray(all_predictions_uncertainties_yaw, dtype=float)  # [N]
                U_ax = np.asarray(all_predictions_uncertainties_ax, dtype=float)  # [N]
                U = U_yaw + U_ax  # e.g., sum; could also be np.maximum(U_yaw, U_ax) or weighted sum

                # Sanitize NaNs/inf so Normalize doesn't explode
                if not np.all(np.isfinite(U)):
                    finite = np.isfinite(U)
                    if finite.any():
                        U = np.where(finite, U, np.nanmean(U) if np.isfinite(np.nanmean(U)) else 0.0)
                    else:
                        U = np.zeros_like(U)

                u_min, u_max = float(np.min(U)), float(np.max(U))
                if np.isclose(u_min, u_max):
                    # Constant uncertainties -> neutral mid value for all
                    norm_val = 0.5
                    norm_fn = None
                else:
                    norm_fn = mcolors.Normalize(vmin=u_min, vmax=u_max)

                cmap_vir = cm.get_cmap("viridis")  # choose any matplotlib colormap

                cmap = cm.get_cmap("tab20")
                # Draw each run with its uncertainty-mapped color
                for i, run in enumerate(self.all_predictions):
                    u = norm_val if norm_fn is None else float(norm_fn(U[i]))
                    if PLOT_UNCERTAINTY:
                        color = cmap_vir(u)[:3]  # RGBA -> RGB tuple
                    else:
                        color = cmap(i % 20)  # RGB A -> RGB tuple

                    self._draw_run(img, run, color=color, goal_point_color=(1.0, 0.1, 1.0))

                # Optional: attach a colorbar to decode uncertainty scale
                # Optional: attach a colorbar to decode uncertainty scale
                if norm_fn is not None and PLOT_UNCERTAINTY:
                    # remove previous colorbar (its axes survives ax.clear())
                    if self._uncert_cbar is not None:
                        try:
                            self._uncert_cbar.remove()  # mpl >= 3.6
                        except Exception:
                            # fallback for older mpl: remove its axes
                            if hasattr(self._uncert_cbar, "ax"):
                                self._uncert_cbar.ax.remove()
                        self._uncert_cbar = None

                    sm = cm.ScalarMappable(norm=norm_fn, cmap=cmap)
                    self._uncert_cbar = self.fig.colorbar(sm, ax=self.ax, fraction=0.046, pad=0.04)
                    self._uncert_cbar.set_label("Uncertainty (Hnorm_yaw + Hnorm_ax)")

                self.ax.imshow(img)

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
            L = lengths[veh]
            pts = np.stack([Xp[veh,:L], Yp[veh,:L]], axis=-1)
            pix = ((pts+center)*res).astype(int)
            pix[:,1] = H-pix[:,1]
            cv2.polylines(img, [pix.reshape(-1,1,2)], False, color,
                          thickness=4)

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

        # Mouse click handler
        def on_click(event):
            if event.inaxes != self.ax:
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

        # Key handler to reset
        def on_key(event):
            if event.key == 'r':
                print("Resetting predictions (press click again to sample).")
                self.all_predictions.clear()
                plot_all()

        # Initial draw
        plot_all()

        # Create temperature slider
        plt.subplots_adjust(bottom=0.15)
        ax_temp = plt.axes([0.25, 0.05, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        temp_slider = Slider(
            ax=ax_temp,
            label='Temperature',
            valmin=0.0,
            valmax=5.0,
            valinit=self.temperature,
            valstep=0.5,
        )

        # Scenario slider
        ax_scenario = plt.axes([0.1, 0.01, 0.35, 0.03], facecolor='lightgoldenrodyellow')
        scenario_slider = Slider(
            ax=ax_scenario,
            label='Scenario',
            valmin=1,
            valmax=len(all_samples),
            valinit=self.current_scenario_idx + 1,
            valstep=1,
        )

        # Vehicle slider
        ax_vehicle = plt.axes([0.55, 0.01, 0.35, 0.03], facecolor='lightgoldenrodyellow')
        vehicle_slider = Slider(
            ax=ax_vehicle,
            label='Vehicle',
            valmin=1,
            valmax=10,  # Set to max vehicles in scenario, you can dynamically detect it too
            valinit=self.current_vehicle_idx + 1,
            valstep=1,
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

        # 2. Make an axes for the radio buttons (position [left, bottom, width, height])
        ax_mode = plt.axes([0.80, 0.05, 0.15, 0.15], facecolor='lightgoldenrodyellow')

        # 3. Create the RadioButtons widget
        mode_radio = RadioButtons(ax_mode, MODES, active=MODES.index(self.sample_mode))

        # 4. Attach a callback
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

        print("Click to sample trajectories; drag the slider to change temperature; press 'r' to clear; close window to exit.")
        plt.show()

        # cleanup
        plt.gcf().canvas.mpl_disconnect(cid_click)
        plt.gcf().canvas.mpl_disconnect(cid_key)
        print("Done.")
        return
