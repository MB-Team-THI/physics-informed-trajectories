import torch
from torch import nn


class diffusion_decoder(nn.Module):
    # ──────────────────────────────────────────────────────────────────
    # 1. ctor – remember input_size so we can use it at inference time
    # ──────────────────────────────────────────────────────────────────
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            *,
            num_diffusion_steps: int = 100,
            use_condition: bool = False
        ):
        super().__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.num_steps   = num_diffusion_steps
        self.use_condition = use_condition
        self.projector = nn.Linear(input_size, output_size)

        # --- εθ --------------------------------------------------------
        self.eps_theta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        # --- β-schedule ----------------------------------------------
        betas = torch.linspace(1e-4, 2e-2, num_diffusion_steps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

        self.generated = None         # inference cache
        self._target_length = None    # remember how many steps we sampled
    # def eps_theta(self):
    #     pass

    # ------------------------------------------------------------------
    # helper: q(x_t | x_0)
    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        sqrt_ab  = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        sqrt_1ab = (1.0 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1ab * noise

    # ------------------------------------------------------------------
    # helper: vanilla DDPM reverse loop – ultra-lean
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _p_sample_loop(self, B, T, device, h_enc, start_frame):
        """
        Generate a (B,T,input_size) trajectory by running the DDPM
        reverse process.  Conditioning is *ignored* for now (same as
        training) – it can be wired in later without changing the API.
        """
        x_t = torch.randn(B, T, self.hidden_size, device=device)  #  ← FIX
        if self.use_condition and start_frame is not None:
            start_frame = start_frame.expand(-1, T, -1)

        for t_now in reversed(range(self.num_steps)):
            # εθ(x_t) – identical to training side
            eps_theta = self.eps_theta(x_t)

            beta      = self.betas[t_now]
            alpha     = 1.0 - beta
            alpha_bar = self.alphas_cumprod[t_now]

            coef1 = 1.0 / alpha.sqrt()
            coef2 = beta / (1.0 - alpha_bar).sqrt()

            x_t = coef1 * (x_t - coef2 * eps_theta)
            if t_now > 0:                        # add noise except at t=0
                x_t = x_t + beta.sqrt() * torch.randn_like(x_t)
        return x_t                               # (B,T,input_size)


    # ------------------------------------------------------------------
    # forward  (identical signature to lstm_decoder)
    # ------------------------------------------------------------------
    def sample(
            self,
            x_start: torch.Tensor,  # (B,1,*)   – frame at current step
            encoder_hidden_states,  # (B,H)
            seq_timestep: int,  # which step the caller wants
            conditions=None
    ):
        """
        Called step-by-step by the kinematic model.  On the first call
        (seq_timestep==0) we draw an entire trajectory once, project it
        to the control space, and cache it.  Later calls just return the
        cached slice.
        """
        # On a *new* roll-out (t==0) drop any stale cache
        if seq_timestep == 0:
            self.generated = None

        # ----------------------------------------------------------------
        # build the cache if we haven't done so for this roll-out
        # ----------------------------------------------------------------
        if self.generated is None:
            B, device = x_start.size(0), x_start.device
            target_len = self._target_length or 30  # keep default, or set
            # externally if you like

            # unpack encoder state exactly like in the training pass
            h_enc = encoder_hidden_states[0] if isinstance(
                encoder_hidden_states, tuple) else encoder_hidden_states
            if h_enc.dim() == 3:
                h_enc = h_enc.squeeze(0)

            start_frame = x_start if self.use_condition else None

            # run DDPM reverse process in *input* space (dim = input_size)
            traj_in_input_space = self._p_sample_loop(
                B, target_len, device, h_enc, start_frame
            )  # (B,T,input_size)

            # map to control/output space so the caller sees the same
            # dimensionality as during training
            self.generated = self.projector(traj_in_input_space)  # (B,T,output_size)
            self._target_length = target_len

        # ----------------------------------------------------------------
        # boundary check & serve requested slice
        # ----------------------------------------------------------------
        if seq_timestep >= self.generated.size(1):
            raise IndexError(
                f"seq_timestep {seq_timestep} exceeds cached length "
                f"{self.generated.size(1)}"
            )

        step_out = self.generated[:, seq_timestep:seq_timestep + 1, :]  # (B,1,D_out)
        return step_out, 1  # second return kept as trivial placeholder

    def forward(
            self,
            x_start: torch.Tensor,  # (B, T, output_size)
            encoder_hidden_states,  # (B, H)
            seq_timestep: int,  # ignored in training
            conditions=None  # optionally (B, 1, input_size)
    ):
        """
        Training forward pass: Sample timestep t, generate x_t with noise,
        predict ε_theta, and return components for loss computation.
        """
        if not self.training:
            return self.sample(x_start, encoder_hidden_states, seq_timestep, conditions)
        B, T, D = x_start.shape
        device = x_start.device

        # --- encode hidden state (B, H)
        if isinstance(encoder_hidden_states, tuple):
            h_enc = encoder_hidden_states[0].squeeze(0)
        else:
            h_enc = encoder_hidden_states
        if h_enc.dim() == 3:
            h_enc = h_enc.squeeze(0)

        # --- Sample timestep t ~ U(0, T)
        t = torch.randint(0, self.num_steps, (B,), device=device, dtype=torch.long)

        # --- Sample noise ε
        noise = torch.randn_like(x_start)

        # --- Noising step: x_t = q(x_t | x_0)
        x_t = self.q_sample(x_start, t, noise)

        # --- Conditioning
        if conditions is not None and False:
            start_frame = conditions.expand(-1, T, -1)  # (B, T, input_size)
            h_expanded = h_enc.unsqueeze(1).expand(-1, T, -1)  # (B, T, H)
            cond = torch.cat([x_t, h_expanded, start_frame], dim=-1)
        else:
            cond = torch.cat([h_enc], dim=-1)  # (B, T, input + hidden)

        # --- Predict ε_θ
        pred_noise = self.eps_theta(h_enc)  # (B, T, D)
        out = self.projector(pred_noise)
        return out, 1 # 183,1,2