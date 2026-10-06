import torch

from src.model.model import model
from .diffusion import diffusion_decoder
from .transformer_decoder import DecoderWithEncoderCrossAttention
import torch.nn.functional as F
from src.model.model_730_utils import *
from src.model.model_101 import model_101
from einops import rearrange
from torch.special import digamma, gammaln

encoderI_types = ["ViT", "ResNet-18"]
encoderT_types = ["LSTM", "Transformer-Encoder"]
merge_types = ["Transformer"]
decoder_types = ["LSTM", "Diffusion", "Transformer"]

EMBEDDING_DIM = 2
NUM_BINS = 128
SAMPLING_MODE = 'gumbel'  # 'continuous' or 'gumbel' or 'deterministic', 'softmax'
GUMBEL_HARD = True
USE_FULL_HISTORY = True
USE_TEACHER_FORCING = False
INFERENCE_TEMPERATURE = 1.0
INFERENCE_SAMPLING_MODE = 'topk'  # 'argmax', 'topk', 'topp', or 'multinomial'
SHUFFLE_BINS = False
SEED = 42
COMPUTE_UNCERTAINTY_METRICS = True

Z_DIM = 128
MERGER_DIM = 256
LAYERS = 4
HEADS = 8
CONDITION_DIM = 5


Z_DIM_I = Z_DIM
Z_DIM_T = Z_DIM - CONDITION_DIM
Z_DIM_IN = Z_DIM
Z_DIM_OUT = MERGER_DIM

DECODER_DIM = Z_DIM_OUT
USE_EVIDENTIAL_HEAD = False
USE_KV_CACHE = False

# Create original bins
YAW_BINS = torch.linspace(-1.5, 1.5, steps=NUM_BINS)
AX_BINS = torch.linspace(-6.0, 3.0, steps=NUM_BINS)
EPS = 1e-8




if SHUFFLE_BINS:
    torch.manual_seed(SEED)
    perm = torch.randperm(NUM_BINS)
    YAW_BINS = YAW_BINS[perm]

    torch.manual_seed(SEED + 1)  # Different seed to get different shuffle
    perm = torch.randperm(NUM_BINS)
    AX_BINS = AX_BINS[perm]


if SAMPLING_MODE == 'continuous':
    NUM_BINS = 2
if SAMPLING_MODE == "deterministic":
    NUM_BINS = 1


def normalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return (t - val_min) / (val_max - val_min)


def denormalize(t: torch.Tensor):
    val_max = 211.625
    val_min = -137.875
    return t * (val_max - val_min) + val_min


def sample_from_bins(probs, bins, mode="multinomial", top_k=10, top_p=0.9, invert_probs=False):
    """
    Samples values from bins using a specified sampling mode.

    Args:
        probs (Tensor): [B, NUM_BINS] softmaxed probabilities
        bins (Tensor): [NUM_BINS] bin centers
        mode (str): 'argmax', 'topk', 'topp', or 'multinomial'
        top_k (int): used if mode is 'topk'
        top_p (float): used if mode is 'topp'

    Returns:
        Tensor: [B] sampled bin values
    """
    if invert_probs and False:
        inv_probs = 1.0 / (probs + 1e-8)  # avoid division by zero
        probs = inv_probs / inv_probs.sum(dim=-1, keepdim=True)  # re-normalize
    B, _ = probs.shape
    if mode == "argmax":
        idx = probs.argmax(dim=-1)
        return bins[idx]
    elif mode == "topk":
        return top_k_sampling(probs, bins, k=top_k)
    elif mode == "topp":
        return top_p_sampling(probs, bins, p=top_p)
    elif mode == "multinomial":
        idx = torch.multinomial(probs, 1).squeeze(-1)
        return bins[idx]
    elif mode == "random":
        idx = torch.randint(low=0, high=NUM_BINS, size=(B,), device=probs.device)
        return bins[idx]
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")

class EvidentialHead(nn.Module):
    """Predict non-negative evidence for a K-way categorical (Dirichlet)."""

    def __init__(self, d_model: int, num_bins: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_bins)

    def forward(self, h):  # h: [B, d_model]
        evidence = F.softplus(self.proj(h))  # non-negative
        alpha = evidence + 1.0  # Dirichlet params
        S = alpha.sum(dim=-1, keepdim=True)  # total evidence (concentration)
        p = alpha / S  # Dirichlet mean (categorical probs)
        return p, alpha, S.squeeze(-1)  # p:[B,K], alpha:[B,K], S:[B]

class EvidentialRegressionHead(nn.Module):
    """Predict NIG params for 1-D regression target: (mu, lam, alpha, beta)."""
    def __init__(self, in_dim,
                 scale_lam: float = 0.1,
                 scale_alpha: float = 0.5,
                 scale_beta: float = 0.1,
                 max_lam: float = 1e2,
                 max_alpha: float = 50.0,
                 max_beta: float = 1e2):
        super().__init__()
        self.fc = nn.Linear(in_dim, 4)
        # conservative bias so we don't start overconfident
        nn.init.constant_(self.fc.bias, -1.0)

        self.scale_lam   = scale_lam
        self.scale_alpha = scale_alpha
        self.scale_beta  = scale_beta
        self.max_lam     = max_lam
        self.max_alpha   = max_alpha
        self.max_beta    = max_beta

    def forward(self, h):
        mu, z_lam, z_alpha, z_beta = self.fc(h).chunk(4, dim=-1)
        # tempered softplus (beta=0.5) tames early gradients
        lam   = F.softplus(z_lam,   beta=0.5) * self.scale_lam   + 1e-6
        alpha = F.softplus(z_alpha, beta=0.5) * self.scale_alpha + 1.0
        beta  = F.softplus(z_beta,  beta=0.5) * self.scale_beta  + 1e-6
        # hard caps to avoid Γ/ln explosions
        lam   = lam.clamp_max(self.max_lam)
        alpha = alpha.clamp_max(self.max_alpha)
        beta  = beta.clamp_max(self.max_beta)
        return mu, lam, alpha, beta
def dirichlet_nll(alpha, y_idx):
    """Expected cross-entropy under Dir(α): ψ(Σα) - ψ(α_y)."""
    S = alpha.sum(dim=-1)                                   # [B]
    a_y = alpha.gather(-1, y_idx.unsqueeze(-1)).squeeze(-1) # [B]
    return digamma(S) - digamma(a_y)                        # [B]

def kl_dirichlet_to_uniform(alpha):
    """KL[Dir(α) || Dir(1)] (penalize unwarranted certainty)."""
    K = alpha.size(-1)
    S = alpha.sum(dim=-1)
    term1 = gammaln(S) - gammaln(alpha).sum(dim=-1)
    term2 = -gammaln(torch.tensor(K, device=alpha.device))
    term3 = ((alpha - 1) * (digamma(alpha) - digamma(S).unsqueeze(-1))).sum(dim=-1)
    return term1 + term2 + term3

def evidential_loss(alpha, y_idx, beta_kl):
    return dirichlet_nll(alpha, y_idx) + beta_kl * kl_dirichlet_to_uniform(alpha).clamp_min(0.)


class EncoderFull(nn.Module):
    def __init__(self, *, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[0], merge_type=merge_types[0],
                 decoder_type=decoder_types[0], encoderI_args=None, encoderT_args=None,
                 z_dim_t=64, z_dim_i=64, zm_dim_in=64, zm_dim_out=128, m_depth=6, m_heads=8, image_size=200, channels=5,
                 traj_size=3, use_infra=True,
                 use_infra_merge=True, dynamic_model='decoupled_dynamic', inference=False):
        super().__init__()
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        assert decoder_type in decoder_types, 'Decoder keyword unknown'
        condition_dim = CONDITION_DIM
        z_dim_i = Z_DIM_I
        z_dim_t = Z_DIM_T
        zm_dim_in = Z_DIM_IN
        zm_dim_out = Z_DIM_OUT
        self.encoderI_type = encoderI_type
        self.encoderT_type = encoderT_type
        self.merge_type = merge_type
        self.decoder_type = decoder_type
        self.use_infra = use_infra
        self.use_infra_merge = use_infra_merge
        self.dynamic_model = dynamic_model
        self.traj_size = traj_size
        self.dim_out_merge = zm_dim_out
        self.inference = inference
        condition_lstm = True



        self.condition_encoder = nn.Linear(2, condition_dim)

        # Image Encoder Infra Encoder==========================================================================================================================
        if self.encoderI_type == encoderI_types[0] and use_infra:
            # ---------------------------------------
            # ViT
            # ---------------------------------------
            if not encoderI_args:
                self.encoder_image = model_101(image_size=image_size, channels=channels, z_dim=z_dim_i, patch_size=20,
                                               dim=256, depth=10, heads=16, mlp_dim=128)
            else:
                self.encoder_image = model_101(image_size=image_size, channels=channels, z_dim=z_dim_i, **encoderI_args)

        elif self.encoderI_type == encoderI_types[1] and use_infra:
            # ---------------------------------------
            # ResNet-18
            # ---------------------------------------
            self.encoder_image = SimpleImageEncoder(channels, z_dim_i)
            # self.encoder_image = models.resnet18()
            # self.encoder_image.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
            #                                      bias=False)
            # self.encoder_image.fc = nn.Linear(512, z_dim_i, bias=True)

        # Trajectory Encoder sq==========================================================================================================================
        if self.encoderT_type == encoderT_types[0]:
            # ---------------------------------------
            # LSTM
            # ---------------------------------------
            self.encoder_trajectory = nn.LSTM(input_size=traj_size, hidden_size=z_dim_t, batch_first=True)
        elif self.encoderT_type == encoderT_types[1]:
            # ---------------------------------------
            # Transformer
            # ---------------------------------------
            if not encoderT_args:
                self.encoder_trajectory = traj_encoder(dim_in=traj_size, dim_out=z_dim_t, depth=6, heads=8,
                                                       dim_trans=64, dim_mlp=128)
            else:
                self.encoder_trajectory = traj_encoder(dim_in=traj_size, dim_out=z_dim_t, **encoderT_args)



        # Encoder Merger==========================================================================================================================
        if self.merge_type == merge_types[0]:
            self.merger_transformer = merger_encoder(dim_in=zm_dim_in, dim_trans=zm_dim_out, depth=m_depth,
                                                     heads=m_heads, condition=False)

            # Decoder Dynamic
        if self.decoder_type == decoder_types[0]:
            if use_infra_merge:  # combine infra hidden also in decoding dynamics
                lstm_decoder_hidden = zm_dim_out * 2
            else:
                lstm_decoder_hidden = zm_dim_out
            if dynamic_model == 'decoupled_dynamic':
                self.deocder_lstm = lstm_decoder(input_size=traj_size, hidden_size=lstm_decoder_hidden,
                                                 output_size=3)  # output vx,vy,t
            elif dynamic_model == 'constant_turn_rate':
                self.deocder_lstm = lstm_decoder(input_size=traj_size, hidden_size=lstm_decoder_hidden,
                                                 output_size=2,
                                                 use_condition=condition_lstm)  # output t,v,ax,psi,psi_dot
            elif dynamic_model == 'deep_kinematic_model':
                self.deocder_lstm = lstm_decoder(input_size=traj_size, hidden_size=lstm_decoder_hidden,
                                                 output_size=3)  # output a,theta,t
        elif self.decoder_type == "Diffusion":  # ------------------ NEW
            if use_infra_merge:
                dec_hidden = zm_dim_out * 2
            else:
                dec_hidden = zm_dim_out

            if dynamic_model == 'decoupled_dynamic':
                out_dim = 3  # vx, vy, t
            elif dynamic_model == 'constant_turn_rate':
                out_dim = 2  # yaw_rate, ax   (t is separate)
            elif dynamic_model == 'deep_kinematic_model':
                out_dim = 3  # a, theta, t
            else:
                raise ValueError(f"Unknown dynamic model: {dynamic_model}")

            self.deocder_lstm = diffusion_decoder(
                input_size=traj_size,
                hidden_size=dec_hidden,
                output_size=out_dim,
                use_condition=condition_lstm  # same flag as before
            )
        else:
            # self.autoregressive_decoder = TrajectoryTransformerDecoder(out_dim=NUM_BINS, sampling_mode=SAMPLING_MODE)
            self.autoregressive_decoder = DecoderWithEncoderCrossAttention(embed_dim=DECODER_DIM, max_seq_len=128,
                                                                     vocab_size=NUM_BINS, num_layers=LAYERS, num_heads=HEADS,
                                                                     dropout=0.1)

    def _init_prediction_buffers(self, B, T, device):
        return (
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device),
            torch.zeros(B, T, device=device)
        )

    def _make_tgt_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), 1).bool()

    def _make_action_sample(self, dec_out, B, temperature, yaw_bins, ax_bins, inference, t, sample_mode_override=None):
        mode = "gumbel"
        dec_t = dec_out

        temperature = max(1e-5, temperature)
        sample_mode = sample_mode_override if sample_mode_override is not None else INFERENCE_SAMPLING_MODE
        invert_probs = False
        if t < 7:
            invert_probs = False

        logits = dec_t.view(B, 2, NUM_BINS)
        tau = temperature

        if self.training:
            yaw_oh = F.gumbel_softmax(logits[:, 0], tau=tau, hard=GUMBEL_HARD)
            ax_oh = F.gumbel_softmax(logits[:, 1], tau=tau, hard=GUMBEL_HARD)
            yaw_sampled = yaw_oh @ yaw_bins
            ax_sampled = ax_oh @ ax_bins
        else:
            yaw_probs = F.softmax(logits[:, 0] / tau, dim=-1)
            ax_probs = F.softmax(logits[:, 1] / tau, dim=-1)
            yaw_sampled = sample_from_bins(yaw_probs, yaw_bins, mode=sample_mode, invert_probs=invert_probs)
            ax_sampled = sample_from_bins(ax_probs, ax_bins, mode=sample_mode, invert_probs=invert_probs)

        return torch.stack((yaw_sampled, ax_sampled), dim=-1)


    def _integrate_motion_model(self, X_n, Y_n, T_n, psi_n, v_n, ax_n, yaw_rate_n, dt, use_second_order):
        cos, sin = torch.cos(psi_n), torch.sin(psi_n)
        if use_second_order:
            half_dt2 = 0.5 * dt * dt
            X_next = X_n + v_n * cos * dt + half_dt2 * (ax_n * cos - v_n * yaw_rate_n * sin)
            Y_next = Y_n + v_n * sin * dt + half_dt2 * (ax_n * sin + v_n * yaw_rate_n * cos)
        else:
            X_next = X_n + v_n * cos * dt
            Y_next = Y_n + v_n * sin * dt

        v_next = v_n + ax_n * dt
        psi_next = psi_n + yaw_rate_n * dt
        T_next = T_n + dt
        return X_next, Y_next, T_next, v_next, psi_next

    def _update_decoder_inputs(self, y_traj, pred_token, t, device):
        if y_traj is not None:
            gt_token = y_traj[:, t, :5].to(device)
            return gt_token
        return pred_token

    def constant_turn_rate_ax_transformer(self, decoder_input, target_length, h_merged,
                                          decoder_input_pack_padded, decoder_hidden,
                                          inference: bool = False,
                                          conditions=None,
                                          temperature: float = 0.1,
                                          y_traj=None,
                                          use_second_order: bool = True,
                                          sample_mode_override=None):

        device = decoder_input.device
        B = decoder_input.size(0)
        if not USE_TEACHER_FORCING or y_traj is None:
            y_traj = None
        else:
            y_traj = torch.nan_to_num(y_traj, nan=0.0, posinf=0.0, neginf=0.0)

        # Init buffers
        X, Y, T, v, psi, psi_d, ax = self._init_prediction_buffers(B, target_length, device)

        # Initial recurrent state
        X_n, Y_n, T_n, psi_n, v_n = decoder_input.t()
        decoder_hidden = decoder_hidden[0]  # [1,B,H]
        yaw_bins = YAW_BINS.to(device)
        ax_bins = AX_BINS.to(device)

        generated_seq = decoder_input_pack_padded  # [B,20,D] # cond is D,B
        conditions = conditions.unsqueeze(1)
        #generated_seq = torch.cat([conditions, generated_seq], dim=1)

        mu_X_steps, mu_Y_steps = [], []
        lam_X_steps, alpha_X_steps, beta_X_steps = [], [], []
        lam_Y_steps, alpha_Y_steps, beta_Y_steps = [], [], []
        logits_acc = []

        for t in range(target_length):
            inp = generated_seq[:, :, :].transpose(0, 1)
            logits = self.autoregressive_decoder(
                inputs_embeds=inp,
                encoder_seq=h_merged,
            ).transpose(0, 1)
            # take logits of last element
            logits = logits[-1]
            if COMPUTE_UNCERTAINTY_METRICS:
                logits_acc.append(logits.detach())
            action = self._make_action_sample(
                logits, B, temperature, yaw_bins, ax_bins, inference, t, sample_mode_override
            )

            raw_yaw_rate, raw_ax = action.t()

            yaw_rate_n, ax_n = raw_yaw_rate, raw_ax

            dt = torch.full_like(yaw_rate_n, 0.1) if inference else 0.1

            X_next, Y_next, T_next, v_next, psi_next = self._integrate_motion_model(
                X_n, Y_n, T_n, psi_n, v_n, ax_n, yaw_rate_n, dt, use_second_order
            )
            v_next = v_next.clamp(min=0.0)

            # Used for calculating the loss
            X[:, t], Y[:, t], T[:, t] = X_next, Y_next, T_next
            v[:, t], psi[:, t] = v_next, psi_next
            psi_d[:, t], ax[:, t] = yaw_rate_n, ax_n

            mu_X_steps.append(X_next)  # [B]
            mu_Y_steps.append(Y_next)

            pred_token = torch.stack((X_next, Y_next, T_next, psi_next, v_next), dim=1)

            next_token = self._update_decoder_inputs(y_traj, pred_token, t, device)
            generated_seq = torch.cat((generated_seq, next_token.unsqueeze(1)), dim=1)


            # next state for the kinematic model
            X_n, Y_n, T_n, psi_n, v_n = X_next, Y_next, T_next, psi_next, v_next

                # check if contains nan

        if COMPUTE_UNCERTAINTY_METRICS:
            logits_acc = torch.stack(logits_acc, dim=0)  # seq, n, (NUM_BINS YAW + NUM_BINS AX
            uncertainty_metrics = compute_uncertainty_metrics(logits_acc, NUM_BINS, NUM_BINS, YAW_BINS, AX_BINS)
        else:
            uncertainty_metrics = {}

        return {
            "X": X, "Y": Y, "T": T, "v": v, "psi": psi,
            "psi_dot": psi_d, "ax": ax, "uncertainty_metrics": uncertainty_metrics,

        }

    # def constant_turn_rate_ax_transformer_kv_cache(self, decoder_input, target_length, h_merged,
    #                                       output_seq_X, # [B,20,D]
    #                                       _,
    #                                       inference: bool = False,
    #                                       conditions=None,
    #                                       temperature: float = 0.1,
    #                                       output_seq_Y=None, # [B,T,D]
    #                                       use_second_order: bool = True,
    #                                       sample_mode_override=None):
    #
    #     device = decoder_input.device
    #     B = decoder_input.size(0)
    #     if not USE_TEACHER_FORCING or output_seq_Y is None:
    #         output_seq_Y = None
    #     else:
    #         output_seq_Y = torch.nan_to_num(output_seq_Y, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #     # Init buffers
    #     X, Y, T, v, psi, psi_d, ax = self._init_prediction_buffers(B, target_length, device)
    #
    #     # Initial recurrent state
    #     X_n, Y_n, T_n, psi_n, v_n = decoder_input.t()
    #     yaw_bins = YAW_BINS.to(device)
    #     ax_bins = AX_BINS.to(device)
    #
    #     generated_seq = output_seq_X  # [B,20,D] # cond is D,B
    #     conditions = conditions.unsqueeze(1)
    #     # generated_seq = torch.cat([conditions, generated_seq], dim=1)
    #
    #     mu_X_steps, mu_Y_steps = [], []
    #     lam_X_steps, alpha_X_steps, beta_X_steps = [], [], []
    #     lam_Y_steps, alpha_Y_steps, beta_Y_steps = [], [], []
    #     logits_acc = []
    #
    #
    #
    #     enc_kv_per_layer, enc_mask, enc_proj = self.autoregressive_decoder.precompute_encoder_kv(h_merged)
    #
    #     # Initialize per-layer past cache
    #     num_layers = len(self.autoregressive_decoder.blocks)
    #     past_self = [None] * num_layers
    #
    #     # If you already have a seed prefix in `generated_seq` (e.g., initial 20 tokens),
    #     # "prime" the cache by stepping through them once (no sampling).
    #     B = generated_seq.size(0)
    #     L0 = generated_seq.size(1)
    #     for t0 in range(L0):
    #         pos_idx = torch.full((B, 1), t0, device=device, dtype=torch.long)
    #         _logits, past_self = self.autoregressive_decoder.step(
    #             new_token_5d=generated_seq[:, t0, :],  # [B,5]
    #             pos_idx=pos_idx,
    #             past_self=past_self,
    #             enc_kv_per_layer=enc_kv_per_layer,
    #             enc_key_padding_mask=enc_mask
    #         )
    #
    #     for t in range(target_length):
    #         pos_idx = torch.full((B, 1), L0 + t, device=device, dtype=torch.long)
    #
    #         # 1) run one decoder step; logits are only for the last token
    #         logits, past_self = self.autoregressive_decoder.step(
    #             new_token_5d=generated_seq[:, -1, :],  # last token as input
    #             pos_idx=pos_idx,
    #             past_self=past_self,
    #             enc_kv_per_layer=enc_kv_per_layer,
    #             enc_key_padding_mask=enc_mask
    #         )
    #         logits = logits[:, -1, :]
    #
    #         if COMPUTE_UNCERTAINTY_METRICS:
    #             logits_acc.append(logits.detach())
    #         action, evi = self._make_action_sample(
    #             logits, B, temperature, yaw_bins, ax_bins, inference, t, sample_mode_override
    #         )
    #
    #         raw_yaw_rate, raw_ax = action.t()
    #
    #         yaw_rate_n, ax_n = raw_yaw_rate, raw_ax
    #
    #         dt = torch.full_like(yaw_rate_n, 0.1) if inference else 0.1
    #
    #         X_next, Y_next, T_next, v_next, psi_next = self._integrate_motion_model(
    #             X_n, Y_n, T_n, psi_n, v_n, ax_n, yaw_rate_n, dt, use_second_order
    #         )
    #         v_next = v_next.clamp(min=0.0)
    #
    #         # Used for calculating the loss
    #         X[:, t], Y[:, t], T[:, t] = X_next, Y_next, T_next
    #         v[:, t], psi[:, t] = v_next, psi_next
    #         psi_d[:, t], ax[:, t] = yaw_rate_n, ax_n
    #
    #         mu_X_steps.append(X_next)  # [B]
    #         mu_Y_steps.append(Y_next)
    #         lam_X_steps.append(evi["lam_X"])
    #         alpha_X_steps.append(evi["alpha_X"])
    #         beta_X_steps.append(evi["beta_X"])
    #         lam_Y_steps.append(evi["lam_Y"])
    #         alpha_Y_steps.append(evi["alpha_Y"])
    #         beta_Y_steps.append(evi["beta_Y"])
    #
    #         pred_token = torch.stack((X_next, Y_next, T_next, psi_next, v_next), dim=1)
    #
    #         next_token = self._update_decoder_inputs(output_seq_Y, pred_token, t, device)
    #         generated_seq = torch.cat((generated_seq, next_token.unsqueeze(1)), dim=1)
    #
    #         # next state for the kinematic model
    #         X_n, Y_n, T_n, psi_n, v_n = X_next, Y_next, T_next, psi_next, v_next
    #         # if teacher forcing we override with true next state
    #         if USE_TEACHER_FORCING:
    #             true_next_state = output_seq_Y[:, t, :]
    #             X_n = true_next_state[:, 0]
    #             Y_n = true_next_state[:, 1]
    #             T_n = true_next_state[:, 2]
    #             psi_n = true_next_state[:, 3]
    #             v_n = true_next_state[:, 4]
    #             # check if contains nan
    #     # stack evidential sequences to [B,T]
    #     mu_X_seq = torch.stack(mu_X_steps, dim=1)  # [B,T]
    #     mu_Y_seq = torch.stack(mu_Y_steps, dim=1)  # [B,T]
    #     lam_X_seq = torch.stack(lam_X_steps, dim=1)  # [B,T]
    #     alpha_X_seq = torch.stack(alpha_X_steps, dim=1)  # [B,T]
    #     beta_X_seq = torch.stack(beta_X_steps, dim=1)  # [B,T]
    #     lam_Y_seq = torch.stack(lam_Y_steps, dim=1)  # [B,T]
    #     alpha_Y_seq = torch.stack(alpha_Y_steps, dim=1)  # [B,T]
    #     beta_Y_seq = torch.stack(beta_Y_steps, dim=1)  # [B,T]
    #     if COMPUTE_UNCERTAINTY_METRICS:
    #         logits_acc = torch.stack(logits_acc, dim=0)  # seq, n, (NUM_BINS YAW + NUM_BINS AX
    #         uncertainty_metrics = compute_uncertainty_metrics(logits_acc, NUM_BINS, NUM_BINS,YAW_BINS, AX_BINS)
    #     else:
    #         uncertainty_metrics = {}
    #
    #     return {
    #         "X": X, "Y": Y, "T": T, "v": v, "psi": psi,
    #         "psi_dot": psi_d, "ax": ax,"uncertainty_metrics": uncertainty_metrics,
    #
    #         # evidential seqs for trainer
    #         "mu_X_seq": mu_X_seq, "mu_Y_seq": mu_Y_seq,  # [B,T]
    #         "lam_X_seq": lam_X_seq, "alpha_X_seq": alpha_X_seq, "beta_X_seq": beta_X_seq,
    #         "lam_Y_seq": lam_Y_seq, "alpha_Y_seq": alpha_Y_seq, "beta_Y_seq": beta_Y_seq,
    #     }


    def forward(self,
                x_image=None,
                x_traj=None,
                y_traj=None,
                x_traj_len=None,
                batch_wise_object_lengths_sum=None,
                batch_wise_decoder_input=None,
                target_length=None,
                conditions=None,
                temperature=1.0,
                sample_mode=None,
                n_gens=1,
                ):
        dataloader_batch_size = x_image.shape[0]
        # Image Encoder==========================================================================================================================
        z_image = self.encoder_image(x_image)

        batch_wise_obj_pack_padded = torch.zeros(
            (batch_wise_object_lengths_sum[-1], x_traj[0].shape[2], x_traj[0].shape[3]),
            device=x_image.device)  # valid_objects x max_seq_len x feat_dim
        obj_length_padded = batch_wise_object_lengths_sum  # valid objects sum over batch
        for unp in range(x_traj[0].shape[0]):
            batch_wise_obj_pack_padded[obj_length_padded[unp]:obj_length_padded[unp + 1], :, :] = x_traj[0][unp,
                                                                                                  0:x_traj[1][unp],
                                                                                                  :,
                                                                                                  :]  # take only valid objects from padded objects

        # x_traj[0][torch.arange(x_traj[0].shape[0]),0:x_traj[1],:,:]

        # normalize
        batch_wise_obj_pack_padded = normalize(batch_wise_obj_pack_padded)
        if EMBEDDING_DIM == 2:
            conditions = normalize_except_zero(conditions)

        batch_wise_obj_traj_padded = torch.nn.utils.rnn.pack_padded_sequence(batch_wise_obj_pack_padded, x_traj_len,
                                                                             batch_first=True,
                                                                             enforce_sorted=False)  # traj should be pack padded, DO NOT TOUCH THE DATA HERE
        batch_wise_obj_traj_embedded = self.encoder_trajectory(batch_wise_obj_traj_padded)  #
        if self.encoderT_type == encoderT_types[0]:
            z_obj = batch_wise_obj_traj_embedded[1][0][0, :,
                    :]  # LSTM: returns hidden state, cell state, take only hidden state valid_obj x embedding_dim
        elif self.encoderT_type == encoderT_types[1]:
            z_obj = batch_wise_obj_traj_embedded

        # Initialize the packed conditions tensor with appropriate dimensions
        # Assuming conditions has the same batch_size and feature_dim as h_merged

        conditions_pack_padded = torch.zeros((1, batch_wise_object_lengths_sum[-1], EMBEDDING_DIM),
                                             device=conditions.device)
        for unp in range(conditions.size(0)):  # conditions.size(0) should give you the batch size
            # Determine the start index for the current object in the packed tensor
            start_idx = obj_length_padded[unp]
            # Determine the end index for the current object in the packed tensor
            # If unp is the last index, use the last value from batch_wise_object_lengths_sum
            end_idx = obj_length_padded[unp + 1] if unp + 1 < len(obj_length_padded) else \
                batch_wise_object_lengths_sum[
                    -1]
            # Pack the conditions, slicing according to the actual lengths in x_traj[1]
            conditions_pack_padded[:, start_idx:end_idx, :] = conditions[unp, :x_traj[1][unp], :]
        conditions_pack_padded = self.condition_encoder(conditions_pack_padded).squeeze(0)

        # concat conditions to z_obj
        z_obj = torch.cat((z_obj, conditions_pack_padded), dim=-1)
        # Encoder Merger==========================================================================================================================
        h_merged, obj_len_updated = self.merger_transformer(z_image, z_obj, obj_length_padded, x_traj[1],
                                                            conditions=None)

        h_merged_infra = h_merged[:, :1, :]
        # Remove Infra from h_merged

        if z_image is not None:
            h_merged = h_merged[:, 1:, :]

        # TODO keep only relevant conditions

        # Decoder Dynamics==========================================================================================================================
        decoder_input = batch_wise_decoder_input  # batch_size x max_obj_len x t[-1] (traj)
        decoder_hidden = h_merged
        # decoder_input_pack_padded = torch.nn.utils.rnn.pack_padded_sequence(decoder_input, x_traj[1], batch_first=True, enforce_sorted=False)
        decoder_hidden_pack_padded = torch.zeros((1, batch_wise_object_lengths_sum[-1], self.dim_out_merge),
                                                 device=x_image.device)
        decoder_input_pack_padded = torch.zeros((batch_wise_object_lengths_sum[-1], 1, self.traj_size),
                                                device=x_image.device)  # valid_objects x 1 x traj_dim
        decoder_hidden_pack_infra = torch.zeros((1, batch_wise_object_lengths_sum[-1], self.dim_out_merge),
                                                device=x_image.device)

        full_trajectory_history = rearrange(x_traj[0], "b v seq dim -> (b v) seq dim")
        # Select only the slices that correspond to actual lengths
        trajectory_history_packed = torch.zeros(
            (
            1, batch_wise_object_lengths_sum[-1], full_trajectory_history.size(1), full_trajectory_history.size(2)),
            device=full_trajectory_history.device)

        for unp in range(x_traj[0].shape[0]):  # batch size
            start_idx = obj_length_padded[unp]
            end_idx = obj_length_padded[unp + 1]
            seq_len = x_traj[1][unp]
            pad = full_trajectory_history.size(1) - seq_len  # shift to the right
            trajectory_history_packed[0, start_idx:end_idx, :, :] = full_trajectory_history[start_idx:end_idx,
                                                                    :, :]

        full_trajectory_history = trajectory_history_packed.squeeze(0)

        for unp in range(x_traj[0].shape[0]):
            decoder_hidden_pack_padded[:, obj_length_padded[unp]:obj_length_padded[unp + 1], :] = decoder_hidden[
                                                                                                  unp,
                                                                                                  0:x_traj[1][unp],
                                                                                                  :]
            decoder_input_pack_padded[obj_length_padded[unp]:obj_length_padded[unp + 1], :, :] = decoder_input[unp,
                                                                                                 0:x_traj[1][unp],
                                                                                                 :].unsqueeze(1)
            decoder_hidden_pack_infra[:, obj_length_padded[unp]:obj_length_padded[unp + 1], :] = h_merged_infra[unp,
                                                                                                 :,
                                                                                                 :]
            # do the same with condition




        if self.use_infra_merge:  # combine infra hidden also in decoding dynamics
            decoder_hidden_pack_padded = torch.cat((decoder_hidden_pack_padded, decoder_hidden_pack_infra), dim=2)

        decoder_input = decoder_input_pack_padded.squeeze(1)
        decoder_hidden = (decoder_hidden_pack_padded, decoder_hidden_pack_padded)

        # --- right after you’ve built `batch_wise_object_lengths_sum` for x_traj ---

        # 1) flatten y_traj into (B*max_obj, seq_len, feat_dim)
        if y_traj is not None:
            full_y_history = rearrange(y_traj[0], "b v seq dim -> (b v) seq dim")
            pred_obj_lens = y_traj[1]  # list[int]  (B,)
            B, V = y_traj[0].shape[:2]  # V = max objects in batch

            # total number of valid objects in the whole batch
            num_valid_total = sum(pred_obj_lens)
            y_trajectory_history_packed = torch.zeros(
                (num_valid_total, full_y_history.size(1), full_y_history.size(2)),
                device=full_y_history.device)

            dst_start = 0
            for scene_idx, num_valid_objs in enumerate(pred_obj_lens):
                src_start = scene_idx * V
                src_end = src_start + num_valid_objs  # only the *real* objects
                y_trajectory_history_packed[dst_start:dst_start + num_valid_objs] = \
                    full_y_history[src_start:src_end]
                dst_start += num_valid_objs




        else:
            y_trajectory_history_packed = None
        # otherwise just feed `y_trajectory_history_packed` directly wherever you need it.
        # --- Encoder memory switch (duplicate vs. scene) ---
        enc_representation = "duplicate"  # "duplicate" | "scene"

        enc_scene = h_merged  # (B, S_enc, E_enc)  (infra already removed above)
        B, S_enc, E_enc = enc_scene.shape
        lens = torch.as_tensor(x_traj[1], device=enc_scene.device, dtype=torch.long)  # (B,)




        enc_key_mask_scene = torch.arange(S_enc, device=enc_scene.device)[None, :] >= lens[:, None]  # (B, S_enc)
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        if enc_representation == "duplicate":
            scene_id = torch.repeat_interleave(torch.arange(B, device=enc_scene.device), lens)  # (N_traj,)
            ENCODER_MEM = enc_scene.index_select(0, scene_id)  # (N_traj, S_enc, dec_E)
            enc_len_traj = lens.index_select(0, scene_id)  # (N_traj,)
            ENCODER_KEY_MASK = torch.arange(S_enc, device=enc_scene.device)[None, :] >= enc_len_traj[:,
                                                                                        None]  # (N_traj, S_enc)

            # ENCODER_KEY_MASK = ~ENCODER_KEY_MASK
            # ENCODER_KEY_MASK = torch.zeros_like(ENCODER_KEY_MASK, dtype=torch.bool)

        else:  # "scene"
            ENCODER_MEM = enc_scene  # (B, S_enc, dec_E)
            ENCODER_KEY_MASK = enc_key_mask_scene

        h_merged = ENCODER_MEM,ENCODER_KEY_MASK
        if self.dynamic_model == 'decoupled_dynamic':
            pass
        elif self.dynamic_model == 'constant_turn_rate':


            initial_state = full_trajectory_history if USE_FULL_HISTORY else decoder_input_pack_padded
            if USE_KV_CACHE:
                output = self.constant_turn_rate_ax_transformer_kv_cache(decoder_input, target_length, h_merged, initial_state,
                                                                decoder_hidden, self.inference, conditions_pack_padded,
                                                                temperature=temperature,
                                                                output_seq_Y=y_trajectory_history_packed,
                                                                sample_mode_override=sample_mode)
            else:
                output = self.constant_turn_rate_ax_transformer(decoder_input, target_length, h_merged, initial_state,
                                                                decoder_hidden, self.inference, conditions_pack_padded,
                                                                temperature=temperature,
                                                                y_traj=y_trajectory_history_packed,
                                                                sample_mode_override=sample_mode)
            return output



class model_730(model):
    def __init__(self,
                 encoderI_type=None,
                 encoderT_type=None,
                 merge_type=None,
                 decoder_type=None,
                 encoderI_args=None,
                 encoderT_args=None,
                 merge_args=None,
                 z_dim_t=64,
                 z_dim_i=64,
                 zm_dim_in=64,
                 zm_dim_out=128,
                 image_size=200,
                 channels=5,
                 traj_size=3,
                 use_infra=True,
                 use_infra_merge=True,
                 dynamic_model=True,
                 m_depth=6,
                 m_heads=8,
                 idx=120,
                 inference=False,
                 name='ScenariioBeta',
                 size=None,
                 n_params=None,
                 onlyEgo=False,
                 input_='image_multichannel_vector',
                 output='vector for objections predicted',
                 task='Trjaectory Forecasting',
                 description='Learning behaviour from latent space'
                 ):
        super().__init__(idx, name, size, n_params, input_, output, task, description)

        self.image_size = image_size
        self.zm_dim_in = zm_dim_in
        self.zm_dim_out = zm_dim_out
        self.onlyEgo = onlyEgo
        self.inference = inference
        self.encoder = EncoderFull(encoderI_type=encoderI_type,
                                   encoderT_type=encoderT_type,
                                   merge_type=merge_type,
                                   decoder_type=decoder_type,
                                   encoderI_args=encoderI_args,
                                   encoderT_args=encoderT_args,
                                   z_dim_t=z_dim_t,
                                   z_dim_i=z_dim_i,
                                   zm_dim_in=self.zm_dim_in,
                                   zm_dim_out=self.zm_dim_out,
                                   m_heads=m_heads,
                                   m_depth=m_depth,
                                   image_size=self.image_size,
                                   channels=channels,
                                   traj_size=traj_size,
                                   use_infra=use_infra,
                                   use_infra_merge=use_infra_merge,
                                   dynamic_model=dynamic_model,
                                   inference=self.inference
                                   )
        self.encoder = self.encoder
        total_params = sum(p.numel() for p in self.encoder.parameters())
        total_trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print("Total params in Millions: ", total_params / 1000000)
        print("Total trainable params in Millions: ", total_trainable_params / 1000000)
        for name, submodule in self.encoder.named_children():
            total_params = sum(p.numel() for p in submodule.parameters())
            total_trainable_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            print(f"{name} - Total params: {total_params:,}, Trainable params: {total_trainable_params}")

    def forward(self,
                x_image=None,
                x_traj=None,
                y_traj=None,
                x_traj_len=None,
                batch_wise_object_lengths_sum=None,
                batch_wise_decoder_input=None,
                target_length=None,
                temperature=1.0,
                conditions=None,
                sample_mode=None):
        output_encode = self.encoder(x_image,
                                     x_traj,
                                     y_traj=y_traj,
                                     x_traj_len=x_traj_len,
                                     temperature=temperature,
                                     batch_wise_object_lengths_sum=batch_wise_object_lengths_sum,
                                     batch_wise_decoder_input=batch_wise_decoder_input,
                                     target_length=target_length, conditions=conditions, sample_mode=sample_mode)

        return output_encode


def generate_model(**model_params):
    return model_730(**model_params)


