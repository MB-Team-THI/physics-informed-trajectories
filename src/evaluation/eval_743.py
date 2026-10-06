import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
matplotlib.use("TkAgg")

from src.evaluation.eval import eval
import cv2
from tqdm import tqdm

# ----------------------------------------------------------------------------- helpers
def normalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return (t - val_min) / (val_max - val_min)

# =============================================================================
#   E V A L   7 4 3  – animated rectangles with heading + temp-slider
# =============================================================================
class eval_743(eval):
    """
    Interactive evaluation with animated vehicle rectangles whose orientation
    matches their instantaneous heading.  A temperature slider re-runs the model
    (and the animation) immediately when changed.
    """

    def __init__(self,
                 idx=121,
                 name='Interactive scenario evaluation',
                 input_='y_true,y_pred',
                 output='acc.',
                 visualize=True,
                 onlyEgo=False,
                 dynamic_model='decoupled_dynamic',
                 description='Interactive scenario: animated vehicle playback'):
        super().__init__(idx, name, input_, output, description)
        self.visualize = visualize
        self.onlyEgo   = onlyEgo
        self.dynamic_model = dynamic_model

        # persistent state
        self.fig = self.axs = self.anim = None
        self.slider = None
        self.model = self.dataloader_test = self.device = self.dataset_dict = None
        self.sample_cache = None
        self.current_scenario_idx = 0
        self.current_vehicle_idx  = 0
        self.temperature = 1.0
        self.target_len  = 30

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def _reshape(output, gX, gY, gT, pred_obj_len, pres_sum):
        X, Y, T = output['X'], output['Y'], output['T']
        Xr = torch.empty_like(gX, device=gX.device)
        Yr = torch.empty_like(gY, device=gY.device)
        Tr = torch.empty_like(gT, device=gT.device)
        for u in range(gX.shape[0]):
            sl = slice(pres_sum[u], pres_sum[u + 1])
            Xr[u, :pred_obj_len[u]] = X[sl].unsqueeze(2)
            Yr[u, :pred_obj_len[u]] = Y[sl].unsqueeze(2)
            Tr[u, :pred_obj_len[u]] = T[sl].unsqueeze(2)
        return Xr, Yr, Tr

    # ---------------------------------------------------------------- public
    def __call__(self, model=None, dataloader_test=None, device=None, dataset_dict=None):
        self.model, self.dataloader_test = model, dataloader_test
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_dict = dataset_dict
        return self._evaluate()

    # ---------------------------------------------------------------- evaluation loop
    def _evaluate(self):
        self.model.to(self.device).eval()

        # ------ take a handful of batches into RAM
        samples = []
        for bi, s in enumerate(tqdm(self.dataloader_test(epoch=0))):
            samples.append(s)
            if bi > 50: break
        if not samples:
            print("No data.")
            return

        self.sample_cache = samples[6]  # pick one scenario for demo
        self.current_scenario_idx = 6

        # ------------------- figure / axes
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 14))
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',   self._on_key)

        # temperature slider (bottom left corner of bottom-right axis)
        ax_slider = inset_axes(self.axs[1, 1], width="80%", height="15%",
                               bbox_to_anchor=(0.1, -0.25, 0.8, 0.2),
                               bbox_transform=self.axs[1, 1].transAxes)
        self.slider = Slider(ax_slider, "Temp", 0.05, 20.0,
                             valinit=self.temperature, valstep=0.05)
        self.slider.on_changed(self._on_slider)

        self._run_inference_and_draw()
        plt.show()

    # ---------------------------------------------------------------- ui handlers
    def _on_slider(self, val):
        self.temperature = val
        self._run_inference_and_draw()

    def _on_key(self, ev):
        if ev.key == 'r':
            self._run_inference_and_draw()

    def _on_click(self, ev):
        if ev.inaxes != self.axs[0, 0] or ev.xdata is None:
            return
        pix_bbox   = self.dataset_dict[0]['bbox_pixel']
        meter_bbox = self.dataset_dict[0]['bbox_meter']
        center     = self.dataset_dict[0]['center_meter']
        res        = np.array(pix_bbox) / np.array(meter_bbox)
        H_img      = pix_bbox[1]

        gx = (ev.xdata / res[0]) - center[0]
        gy = (H_img - ev.ydata) / res[1] - center[1]
        print(f"New goal: ({gx:.2f},{gy:.2f})")
        self.sample_cache['cond_goal_point'][0, self.current_vehicle_idx, 0] = gx
        self.sample_cache['cond_goal_point'][0, self.current_vehicle_idx, 1] = gy
        self._run_inference_and_draw()

    # ---------------------------------------------------------------- heavy lifting
    def _run_inference_and_draw(self):
        if self.anim is not None:
            self.anim.event_source.stop()

        s = self.sample_cache
        x_img   = s['images'].to(self.device)
        x_traj  = [s['hist_objs'].to(self.device), s['hist_obj_lens']]
        traj_len_hist = s['hist_objs_seq_len']
        traj_len_pred = s['pred_objs_seq_len']
        traj_pred_obj_len = s['pred_obj_lens']
        pres_sum = s['pres_object_lengths_sum']
        obj_len_pad = s['hist_object_lengths_sum']
        decoder_in  = s['obj_decoder_in'].to(self.device)

        gX, gY, gT = (s['pred_objsx'].to(self.device),
                      s['pred_objsy'].to(self.device),
                      s['pred_objst'].to(self.device))

        cond_goal = s['cond_goal_point'].to(self.device)
        cond_v    = s['cond_v'].to(self.device)
        conditions = torch.cat([cond_goal], dim=2)              # 2-D goal only

        with torch.no_grad():
            out = self.model(x_image=x_img, x_traj=x_traj, x_traj_len=traj_len_hist,
                             batch_wise_object_lengths_sum=obj_len_pad,
                             conditions=conditions,
                             batch_wise_decoder_input=decoder_in,
                             target_length=self.target_len,
                             temperature=self.temperature)

        Xr, Yr, _ = self._reshape(out, gX, gY, gT,
                                  traj_pred_obj_len, pres_sum)

        # ---------------------------------------------------------------- draw everything
        # for ax_row in self.axs:
        #     for ax in ax_row:
        #         ax.cla()

        idx = 0                              # single frame
        bg = cv2.cvtColor(x_img[idx, 0].cpu().numpy(), cv2.COLOR_GRAY2RGB)

        center = self.dataset_dict[0]['center_meter']
        pix_bb = self.dataset_dict[0]['bbox_pixel']
        met_bb = self.dataset_dict[0]['bbox_meter']
        res    = np.array(pix_bb) / np.array(met_bb)
        H_img  = pix_bb[1]

        n_veh   = x_traj[1][idx]
        px, py  = [], []
        headings = []
        for v in range(n_veh):
            Lp = int(traj_len_pred[obj_len_pad[idx] + v])
            xs = Xr[idx, v, :Lp, 0].cpu().numpy()
            ys = Yr[idx, v, :Lp, 0].cpu().numpy()
            pix_x = (xs + center[0]) * res[0]
            pix_y = H_img - (ys + center[1]) * res[1]
            px.append(pix_x)
            py.append(pix_y)
            # compute heading (rad) for each step
            hd = np.zeros_like(xs)
            if Lp >= 2:
                hd[:-1] = np.arctan2(np.diff(ys), np.diff(xs))
                hd[-1]  = hd[-2]
            headings.append(hd)

        # ------------- axis (0,0) : animated rectangles + trail + endpoints
        self.axs[0, 0].imshow(bg); self.axs[0, 0].axis("off")
        self.axs[0, 0].set_title("Predicted motion (temperature = %.2f)" % self.temperature)

        cmap = plt.cm.get_cmap("tab20", n_veh)
        rects, trails, endpoints = [], [], []
        for v in range(n_veh):
            w_pix = 4 * res[0]
            h_pix = 2 * res[1]
            if not len(px[v]): continue
            r = plt.Rectangle((px[v][0]-w_pix/2, py[v][0]-h_pix/2),
                              w_pix, h_pix,
                              linewidth=1,
                              edgecolor=cmap(v),
                              facecolor=cmap(v, alpha=.4))
            self.axs[0, 0].add_patch(r)
            rects.append(r)

            # trail Line2D
            ln, = self.axs[0, 0].plot([], [], color=cmap(v), linewidth=1.5)
            trails.append(ln)

            # endpoint marker (static)
            ep = self.axs[0, 0].scatter(px[v][-1], py[v][-1],
                                        s=30, c=[cmap(v)], marker='x', zorder=5)
            endpoints.append(ep)

        # ------------- animate
        max_frames = 30

        def _step(f):
            for v, r in enumerate(rects):
                L = len(px[v])
                if L == 0: continue
                k = f % L
                w_pix = r.get_width(); h_pix = r.get_height()
                r.set_xy((px[v][k]-w_pix/2, py[v][k]-h_pix/2))
                r.set_angle(np.degrees(headings[v][k]))       # <-- heading
                trails[v].set_data(px[v][:k+1], py[v][:k+1])  # <-- trail
            return rects + trails

        self.anim = animation.FuncAnimation(self.fig, _step,
                                            frames=max_frames,
                                            interval=200,
                                            blit=True,
                                            repeat=True)

        # ------------- quick static velocity plot (0,1)
        if 'v' in out:
            v_pred = out['v'][obj_len_pad[idx]:obj_len_pad[idx]+n_veh].cpu().numpy()
            for v in range(n_veh):
                self.axs[0, 1].plot(v_pred[v], color=cmap(v), label=f'Veh {v+1}')
            self.axs[0, 1].set_title("Predicted velocity"); self.axs[0, 1].legend(); self.axs[0, 1].grid()

        # leave bottom-row right axis for the slider label only
        self.axs[1, 1].set_axis_off()
        self.axs[1, 0].set_axis_off()

        # self.fig.tight_layout()
        self.fig.canvas.draw_idle()

# ------------------------------------------------------------------------------
# Demo / sanity-check
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    dummy_model = nn.Identity()          # plug in your real model
    def dummy_dl(epoch=0):
        batch = {
            'images': torch.zeros((1,1,500,500)),
            'hist_objs': torch.zeros((1,15,20,3)),
            'hist_obj_lens': torch.LongTensor([ [20]*15 ]),
            'hist_objs_seq_len': torch.LongTensor([20]*15),
            'hist_object_lengths_sum': torch.LongTensor([0,15]),
            'pred_objs_seq_len': torch.LongTensor([30]*15),
            'pred_obj_lens': torch.LongTensor([30]*15),
            'pres_object_lengths_sum': torch.LongTensor([0,15]),
            'obj_decoder_in': torch.zeros((1,15,1,3)),
            'pred_objsx': torch.zeros((1,15,30,1)),
            'pred_objsy': torch.zeros((1,15,30,1)),
            'pred_objst': torch.zeros((1,15,30,1)),
            'cond_goal_point': torch.zeros((1,15,2)),
            'cond_v': torch.zeros((1,15,1)),
        }
        yield batch
    dummy_ds_dict = [{
        'center_meter': np.array([0.,0.]),
        'bbox_pixel':   np.array([500,500]),
        'bbox_meter':   np.array([100,100]),
    }]

    ev = eval_743()
    ev(model=dummy_model,
       dataloader_test=dummy_dl,
       device=torch.device("cpu"),
       dataset_dict=dummy_ds_dict)
