#!/usr/bin/env python3
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BicycleDx(torch.nn.Module):
    """
    Kinematic bicycle model API.

    State vector (n_state = 5):
        [ X, Y, cos(theta), sin(theta), v ]

    Control vector (n_ctrl = 2):
        [ a, delta ]

    forward(state, u) accepts:
        state: (batch, 5) or (5,)  -> returns same shape
        u: (batch, 2) or (2,)
    """

    def __init__(self, params=None):
        super().__init__()

        self.n_state = 5
        self.n_ctrl = 2

        # model params: dt and wheelbase L
        if params is None:
            # (dt, L)
            self.params = Variable(torch.Tensor((0.1, 1.0)))
        else:
            self.params = params
        assert len(self.params) == 2
        self.dt = float(self.params[0])
        self.L = float(self.params[1])

        # control limits
        self.accel_lim = 2.0
        self.steer_lim = 0.5  # radians, adjust as needed
        self.goal_speed = 2.5

        # simple goal/cost defaults (match Cartpole style)
        # goal_state in state-coordinates: [X, Y, cos(th), sin(th), v]
        self.goal_state = torch.Tensor([0.0, 0.0, 1.0, 0.0, self.goal_speed])
        self.goal_weights = torch.Tensor([0.0, 0.15, 0.0, 1.0, 0.5])
        self.ctrl_penalty = 0.001

        # solver related (kept for interface parity)
        self.mpc_eps = 1e-4
        self.linesearch_decay = 0.5
        self.max_linesearch_iter = 2

    def forward(self, state, u):
        """
        Propagate state by one dt using kinematic bicycle model.

        state: (batch,5) or (5,)
        u: (batch,2) or (2,)
        """
        squeeze = state.ndimension() == 1

        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        # cast params to device if needed
        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        device = state.device
        dtype = state.dtype

        dt = (
            float(self.params[0].item())
            if isinstance(self.params, torch.Tensor)
            else self.dt
        )
        L = (
            float(self.params[1].item())
            if isinstance(self.params, torch.Tensor)
            else self.L
        )

        # clamp controls per-batch
        a = torch.clamp(u[:, 0], -self.accel_lim, self.accel_lim)
        delta = torch.clamp(u[:, 1], -self.steer_lim, self.steer_lim)

        # unpack state
        X, Y, cos_th, sin_th, v = torch.unbind(state, dim=1)
        th = torch.atan2(sin_th, cos_th)

        # safe steering warp (avoid tan blow-up); adjust clamp if needed
        # compute slip angle beta and its geometry
        beta = torch.atan(0.5 * torch.tan(delta))  # assumes Lf = Lr = L/2

        # kinematic derivatives
        dx = v * torch.cos(th + beta)
        dy = v * torch.sin(th + beta)
        theta_dot = (v / L) * torch.sin(beta)  # yaw rate
        dcos = -sin_th * theta_dot
        dsin = cos_th * theta_dot
        dv = a

        # integrate (forward Euler)
        X = X + dt * dx
        Y = Y + dt * dy
        cos_th = cos_th + dt * dcos
        sin_th = sin_th + dt * dsin
        v = v + dt * dv

        # re-normalize cos/sin occasionally to avoid drift
        norm = torch.sqrt(cos_th * cos_th + sin_th * sin_th + 1e-12)
        cos_th = cos_th / norm
        sin_th = sin_th / norm

        state_next = torch.stack((X, Y, cos_th, sin_th, v), dim=1)

        if squeeze:
            state_next = state_next.squeeze(0)
        return state_next

    def get_true_obj(self):
        """
        Return Quadratic cost weights (q) and linear term (p) in
        the same format cartpole uses: returns (q, p) both Variables
        q: [state_weights, control_weights]
        p: linear term (used to make objective -sqrt(w)*goal in state part)
        """
        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        px = -torch.sqrt(self.goal_weights) * self.goal_state
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)

    def get_frame(self, state, ax=None):
        """
        Simple top-down visualization of vehicle pose.
        state: (5,) or (batch,5) - we take first if batch
        """
        st = state.view(-1).cpu().detach().numpy()
        X, Y, cos_th, sin_th, v = st[:5]
        th = np.arctan2(sin_th, cos_th)

        # vehicle outline
        L = self.L
        heading = np.array([np.cos(th), np.sin(th)])
        left = np.array([-heading[1], heading[0]]) * 0.6 * L
        right = -left

        p_front = np.array([X, Y]) + heading * (0.6 * L)
        p_back = np.array([X, Y]) - heading * (0.4 * L)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        ax.plot([p_back[0], p_front[0]], [p_back[1], p_front[1]], "-k", lw=3)
        ax.scatter(X, Y, s=40, color="red")
        ax.set_aspect("equal", "box")
        lim = L * 3
        ax.set_xlim(X - lim, X + lim)
        ax.set_ylim(Y - lim, Y + lim)
        return fig, ax


if __name__ == "__main__":
    import os
    import imageio

    out_dir = "bicycle_frames"
    os.makedirs(out_dir, exist_ok=True)

    dx = BicycleDx()
    th = 0.2
    x0 = torch.tensor([0.0, 0.0, np.cos(th), np.sin(th), 1.0])
    u_seq = torch.zeros(30, 2)
    u_seq[:, 0] = 0.2  # acceleration
    u_seq[:, 1] = 0.05  # steering

    x = x0
    traj = [x0]

    frames = []
    for t in range(u_seq.shape[0]):
        fig, ax = dx.get_frame(x)
        frame_path = os.path.join(out_dir, f"frame_{t:03d}.png")
        fig.savefig(frame_path)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

        x = dx(x, u_seq[t] + torch.Tensor([t * 0.02, (t % 5) * 0.2]))
        traj.append(x)

    # Create video
    video_path = "bicycle_sim.mp4"
    imageio.mimsave(video_path, frames, fps=10)
    print(f"Saved video to {video_path}")
