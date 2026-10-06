import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,RadioButtons
from tqdm import tqdm, trange
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from scipy.stats import gaussian_kde


UPSAMPLE_FACTOR = 8
import os
MODES = ["argmax", "topk", "topp", "multinomial"]  # whatever your model supports
VIOLIN_BW = 0.2
### WITH Action DISTIRUBTIOn
class eval_783:
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
                 num_samples=50,
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
        self.sample_mode = "topk"

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

    def _compute_mean_and_var(self):
        """
        Returns:
          mean_pts: (T, 2) array of mean [x,y] over samples
          var_norm: (T,) array of normalized variance magnitude in [0,1]
        """
        pts_list = []
        for run in self.all_predictions:
            X = run['X_reshaped'][0].cpu().numpy()[:,:,0]  # shape (veh, T)
            Y = run['Y_reshaped'][0].cpu().numpy()[:,:,0]
            idx = self.current_vehicle_idx
            traj = np.stack([X[idx], Y[idx]], axis=1)     # (T,2)
            pts_list.append(traj)
        all_pts = np.stack(pts_list, axis=0)             # (N,T,2)
        mean_pts = all_pts.mean(axis=0)                  # (T,2)
        var = all_pts.var(axis=0)                        # (T,2)
        var_mag = np.linalg.norm(var, axis=1)            # (T,)
        # normalize to [0,1] for colormap
        vmin, vmax = var_mag.min(), var_mag.max()
        var_norm = (var_mag - vmin) / (vmax - vmin + 1e-6)
        return mean_pts, var_norm

    def _draw_uncertainty(self, ax_map):
        """
        Overlays a heatmap showing spatial uncertainty based on trajectory spread.
        """
        sc = self.all_predictions
        if len(sc) == 0:
            return

        pts_list = []
        for run in sc:
            X = run['X_reshaped'][0].cpu().numpy()[:, :, 0]
            Y = run['Y_reshaped'][0].cpu().numpy()[:, :, 0]
            idx = self.current_vehicle_idx
            pts = np.stack([X[idx], Y[idx]], axis=1)  # shape (T, 2)
            pts_list.append(pts)
        all_pts = np.concatenate(pts_list, axis=0)  # shape (N*T, 2)

        # Convert to pixel space
        center = self.dataset_dict[0]['center_meter']
        res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
        res *= UPSAMPLE_FACTOR
        H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR

        pix_pts = ((all_pts + center) * res)
        pix_pts[:, 1] = H - pix_pts[:, 1]  # flip y-axis

        # KDE
        kde = gaussian_kde(pix_pts.T, bw_method=0.2)
        W, H_img = int(self.dataset_dict[0]['bbox_pixel'][0] * UPSAMPLE_FACTOR), int(H)
        xi, yi = np.mgrid[0:W:1, 0:H_img:1]
        coords = np.vstack([xi.ravel(), yi.ravel()])
        density = kde(coords).reshape(W, H_img).T

        density = density / density.max()  # normalize

        # Overlay on the *same* axis where the BEV is drawn
        ax_map.imshow(density, cmap='hot', interpolation='bilinear', alpha=0.4, origin='upper')

    def _evaluate(self):
        # Prepare model & data
        self.model.eval()
        self.model.to(self.device)
        all_samples = []
        for i, sample in enumerate(tqdm(self.dataloader_test(epoch=0))):
            all_samples.append(sample)
            if i >= 30:
                break
        if not all_samples:
            print("No data in dataloader. Exiting.")
            return

        # Scenario selection
        max_idx = len(all_samples)
        s = "5"#input(f"Enter scenario index [1..{max_idx}] (default=1): ").strip()
        self.current_scenario_idx = max(0, min(max_idx-1, int(s)-1)) if s else 0
        self.sample_cache = all_samples[self.current_scenario_idx]

        # Vehicle selection
        v = "4"#input("Enter single vehicle index [1..N] (default=1): ").strip()
        self.current_vehicle_idx = max(0, int(v)-1) if v else 0

        # Helper to run one forward pass
        def run_once():
            sc = self.sample_cache
            x_image = sc['images'].to(self.device)
            cond = torch.cat([sc['cond_goal_point'].to(self.device).float()], dim=2)
            with torch.no_grad():
                out = self.model(
                    x_image=x_image,
                    x_traj=[sc['hist_objs'].to(self.device), sc['hist_obj_lens']],
                    x_traj_len=sc['hist_objs_seq_len'],
                    batch_wise_object_lengths_sum=sc['hist_object_lengths_sum'],
                    conditions=cond,
                    batch_wise_decoder_input=sc['obj_decoder_in'].to(self.device),
                    target_length=self.target_len,
                    temperature=self.temperature,
                    sample_mode=self.sample_mode
                )
            Xr, Yr, Tr = self._process_dynamic_model(
                out, sc['pred_objsx'].to(self.device),
                     sc['pred_objsy'].to(self.device),
                     sc['pred_objst'].to(self.device),
                     sc['pred_obj_lens'],
                     sc['pres_object_lengths_sum']
            )
            return {
                'X_reshaped': Xr,
                'Y_reshaped': Yr,
                'psi_dot': out['psi_dot'],
                'ax': out['ax'],
                'v': out['v']
            }

        # Plotting routine
        def plot_all():
            if self.fig is None:
                self.fig, self.ax = plt.subplots(1, 2, figsize=(16, 12))

            img = self._get_background_image()

            for ax in self.ax:
                ax.clear()

            # 1. draw BEV image
            self._draw_history(img, color=(.6, .6, .6))
            self.ax[0].imshow(img)
            self.ax[0].axis('off')

            # 2. draw heatmap on top
            self._draw_uncertainty(self.ax[0])  # <=== overlay here!

            # 3. optionally draw samples
            # for i, run in enumerate(self.all_predictions):
            #     self._draw_run(img, run, color=cmap[i % len(cmap)])

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
                L = lengths[veh]
                pts = traj[veh,:L,:2]
                pix = ((pts+center)*res).astype(int)
                pix[:,1] = H-pix[:,1]
                cv2.polylines(img, [pix.reshape(-1,1,2)], False, color,
                              thickness=2*UPSAMPLE_FACTOR)
        self._draw_history = draw_hist
        def draw_dist(img, color, ax):
            sc = self.all_predictions
            if len(sc) == 0:
                return
            idx = 0
            center = self.dataset_dict[0]['center_meter']
            res = np.array(self.dataset_dict[0]['bbox_pixel']) / np.array(self.dataset_dict[0]['bbox_meter'])
            res *= UPSAMPLE_FACTOR
            H = self.dataset_dict[0]['bbox_pixel'][1] * UPSAMPLE_FACTOR

            gen_ax =  [el['ax'][idx].cpu().numpy() for el in sc]
            gen_ax = np.stack(gen_ax, axis=0)
            gen_psi_dot = [el['psi_dot'][idx].cpu().numpy() for el in sc]
            gen_psi_dot = np.stack(gen_psi_dot, axis=0)
            # both of shape (n,30)

            plot_type = "violin"  # or: "box", "scatter", "hist2d", "mean_std", "violin", "density"

            # --- Plot for a_x ---
            if plot_type == "violin":
                # Reshape to long-form DataFrame for Seaborn
                df = pd.DataFrame(gen_ax)
                df_long = df.melt(var_name="Timestep", value_name="a_x")

                sns.violinplot(
                    data=df_long,
                    x="Timestep",
                    y="a_x",
                    ax=ax[1],
                    inner=None,  # No median or quartiles
                    linewidth=0.8,
                    bw=VIOLIN_BW,
                    scale="width"  # Optional: keeps violin width constant
                )

                ax[1].set_ylabel(r"$a_x$")
                ax[1].set_xlabel("Timestep")
                ax[1].set_title(r"$a_x$ Violin Plot (Seaborn)")

            elif plot_type == "box":
                data = [gen_ax[:, t] for t in range(gen_ax.shape[1])]
                ax[1].boxplot(data)
                ax[1].set_xticks(range(1, gen_ax.shape[1] + 1))
                ax[1].set_xticklabels(range(0, gen_ax.shape[1]))
                ax[1].set_ylabel(r"$a_x$")
                ax[1].set_xlabel("Timestep")
                ax[1].set_title(r"$a_x$ Box Plot")

            elif plot_type == "scatter":
                for t in range(gen_ax.shape[1]):
                    ax[1].scatter([t] * gen_ax.shape[0], gen_ax[:, t], alpha=0.2, color='blue', s=5)
                ax[1].set_ylabel(r"$a_x$")
                ax[1].set_xlabel("Timestep")
                ax[1].set_title(r"$a_x$ Scatter Plot")

            elif plot_type == "hist2d":
                ax[1].hist2d(
                    np.tile(np.arange(gen_ax.shape[1]), gen_ax.shape[0]),
                    gen_ax.flatten(),
                    bins=(gen_ax.shape[1], 50),
                    cmap='viridis'
                )
                ax[1].set_ylabel(r"$a_x$")
                ax[1].set_xlabel("Timestep")
                ax[1].set_title(r"$a_x$ 2D Histogram")

            elif plot_type == "mean_std":
                mean = gen_ax.mean(axis=0)
                std = gen_ax.std(axis=0)
                ts = range(gen_ax.shape[1])

                ax[1].plot(ts, mean, color='blue', label='Mean')

                # ±1σ band
                ax[1].fill_between(ts,
                                   mean - std,
                                   mean + std,
                                   color='blue',
                                   alpha=0.3,
                                   label='±1σ')



                ax[1].set_ylabel(r"$a_x$")
                ax[1].set_xlabel("Timestep")
                ax[1].set_title(r"$a_x$ Mean ± Std")
                ax[1].legend(loc='upper right')
            elif plot_type == "density":
                # --- a_x ---
                for traj in gen_ax:
                    ax[1].plot(traj, color='red', alpha=0.02)

                # 2) over‐plot your mean (and maybe your ±std if you still want it)
                mean = gen_ax.mean(axis=0)
                std = gen_ax.std(axis=0)
                ax[1].plot(mean, color='blue', label='Mean')
                ax[1].fill_between(np.arange(gen_ax.shape[1]),
                                mean - std, mean + std,
                                color='blue', alpha=0.3,
                                label='Mean ± Std')

                ax[1].set_xlabel("Timestep")
                ax[1].set_ylabel(r"$a_x$")
                ax[1].legend()

            # --- Plot for psi_dot ---
            if plot_type == "violin":
                # Seaborn violinplot for ψ̇
                df_psi = pd.DataFrame(gen_psi_dot)
                df_psi_long = df_psi.melt(var_name="Timestep", value_name="psi_dot")

                sns.violinplot(
                    data=df_psi_long,
                    x="Timestep",
                    y="psi_dot",
                    ax=ax[2],
                    inner=None,
                    bw=VIOLIN_BW,
                    linewidth=0.8,
                    scale="width"  # options: 'area', 'count', 'width'
                )

                ax[2].set_ylabel(r"$\dot{\psi}$")
                ax[2].set_xlabel("Timestep")
                ax[2].set_title(r"$\dot{\psi}$ Violin Plot (Seaborn)")


            elif plot_type == "box":
                data_psi = [gen_psi_dot[:, t] for t in range(gen_psi_dot.shape[1])]
                ax[2].boxplot(data_psi)
                ax[2].set_xticks(range(1, gen_psi_dot.shape[1] + 1))
                ax[2].set_xticklabels(range(0, gen_psi_dot.shape[1]))
                ax[2].set_ylabel(r"$\dot{\psi}$")
                ax[2].set_xlabel("Timestep")
                ax[2].set_title(r"$\dot{\psi}$ Box Plot")

            elif plot_type == "scatter":
                for t in range(gen_psi_dot.shape[1]):
                    ax[2].scatter([t] * gen_psi_dot.shape[0], gen_psi_dot[:, t], alpha=0.2, color='green', s=5)
                ax[2].set_ylabel(r"$\dot{\psi}$")
                ax[2].set_xlabel("Timestep")
                ax[2].set_title(r"$\dot{\psi}$ Scatter Plot")

            elif plot_type == "hist2d":
                ax[2].hist2d(
                    np.tile(np.arange(gen_psi_dot.shape[1]), gen_psi_dot.shape[0]),
                    gen_psi_dot.flatten(),
                    bins=(gen_psi_dot.shape[1], 50),
                    cmap='viridis'
                )
                ax[2].set_ylabel(r"$\dot{\psi}$")
                ax[2].set_xlabel("Timestep")
                ax[2].set_title(r"$\dot{\psi}$ 2D Histogram")

            elif plot_type == "mean_std":
                mean = gen_psi_dot.mean(axis=0)
                std = gen_psi_dot.std(axis=0)
                ax[2].plot(mean, color='green', label='Mean')
                ax[2].fill_between(range(gen_psi_dot.shape[1]), mean - std, mean + std, alpha=0.3, color='green')
                ax[2].set_ylabel(r"$\dot{\psi}$")
                ax[2].set_xlabel("Timestep")
                ax[2].set_title(r"$\dot{\psi}$ Mean ± Std")

        self._draw_dist = draw_dist

        def draw_run(img, run, color):
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
                          thickness=1*UPSAMPLE_FACTOR)

            gx, gy = sc['cond_goal_point'][0,veh]
            gp = np.array([gx,gy])
            pixg = ((gp+center)*res).astype(int)
            pixg[1] = H-pixg[1]
            cv2.drawMarker(img, tuple(pixg), color,
                           markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=6*UPSAMPLE_FACTOR,
                           thickness=1*UPSAMPLE_FACTOR)
        self._draw_run = draw_run

        # Mouse click handler
        def on_click(event):
            if event.inaxes != self.ax[0]:
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
                self.all_predictions.append(run_once())
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
            valmax=20.0,
            valinit=self.temperature,
            valstep=0.5,
        )

        def on_temp_change(val):
            self.temperature = val
            print(f"Temperature set to {val:.2f}. Re-sampling…")
            self.all_predictions.clear()
            for _ in trange(self.num_samples):
                self.all_predictions.append(run_once())
            plot_all()

        temp_slider.on_changed(on_temp_change)


        # 2. Make an axes for the radio buttons (position [left, bottom, width, height])
        ax_mode = plt.axes([0.80, 0.05, 0.15, 0.15], facecolor='lightgoldenrodyellow')

        # 3. Create the RadioButtons widget
        mode_radio = RadioButtons(ax_mode, MODES, active=MODES.index(self.sample_mode))

        # 4. Attach a callback
        def on_mode_change(label):
            self.sample_mode = label
            print(f"Sample mode set to {label}. Re-sampling…")
            self.all_predictions.clear()
            for _ in range(self.num_samples):
                self.all_predictions.append(run_once())
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
