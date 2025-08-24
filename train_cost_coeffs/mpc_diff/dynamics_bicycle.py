# bicycle_dx.py
import torch
from torch.autograd import Variable
import numpy as np
from torch import nn


class BicycleDx(torch.nn.Module):
    """
    Kinematic bicycle model.

    State (n_state = 5): [X, Y, cos(theta), sin(theta), v]
    Control (n_ctrl = 2): [a, delta]
    """

    def __init__(self, params=None):
        super().__init__()
        self.n_state = 5
        self.n_ctrl = 2

        # params: (dt, L)
        if params is None:
            self.params = Variable(torch.tensor((0.1, 1.0), dtype=torch.float32))
        else:
            self.params = params
        assert len(self.params) == 2

        self.accel_lim = 2.0
        self.steer_lim = 0.5
        self.goal_speed = 2.5

        # Direction-agnostic lane keeping default target:
        #   - No X penalty (free progress)
        #   - Penalize Y -> 0
        #   - Penalize sin(theta) -> 0 (parallel regardless of direction)
        #   - Regulate speed -> goal_speed
        # These are only defaults; training will learn the diagonals.
        self.goal_state = torch.tensor([0.0, 0.0, 1.0, 0.0, self.goal_speed])
        self.goal_weights = torch.tensor([0.0, 0.15, 0.0, 1.0, 0.5])  # diag (state)
        self.ctrl_penalty = 0.001  # diag (control)

    def forward(self, state, u):
        squeeze = state.ndimension() == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        if state.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        dt = float(self.params[0].item())
        L = float(self.params[1].item())

        a = torch.clamp(u[:, 0], -self.accel_lim, self.accel_lim)
        delta = torch.clamp(u[:, 1], -self.steer_lim, self.steer_lim)

        X, Y, cth, sth, v = torch.unbind(state, dim=1)
        th = torch.atan2(sth, cth)

        beta = torch.atan(0.5 * torch.tan(delta))  # Lf= Lr = L/2

        dx = v * torch.cos(th + beta)
        dy = v * torch.sin(th + beta)
        theta_dot = (v / L) * torch.sin(beta)
        dcth = -sth * theta_dot
        dsth = cth * theta_dot
        dv = a

        X = X + dt * dx
        Y = Y + dt * dy
        cth = cth + dt * dcth
        sth = sth + dt * dsth
        v = v + dt * dv

        # renormalize [cos, sin]
        norm = torch.sqrt(cth * cth + sth * sth + 1e-12)
        cth /= norm
        sth /= norm

        xnext = torch.stack((X, Y, cth, sth, v), dim=1)
        if squeeze:
            xnext = xnext.squeeze(0)
        return xnext

    def get_true_obj(self):
        """
        Returns (q, p) as 1D vectors for QuadCost construction (Cartpole-style):
          q: length (n_state + n_ctrl) – diagonal weights only
          p: length (n_state + n_ctrl) – linear term
        By default:
          - penalize Y (idx=1), sin(theta) (idx=3), speed error (idx=4)
          - no X or cos(theta) penalty
          - small control effort penalty on both controls
        """
        q_state = self.goal_weights  # (5,)
        q_ctrl = self.ctrl_penalty * torch.ones(self.n_ctrl)  # (2,)
        q = torch.cat((q_state, q_ctrl), dim=0)

        # linear term: p_state = -sqrt(q_state) * goal_state (Cartpole convention)
        sqrt_q_state = torch.sqrt(torch.clamp(q_state, min=0))
        p_state = -sqrt_q_state * self.goal_state
        p_ctrl = torch.zeros(self.n_ctrl)
        p = torch.cat((p_state, p_ctrl), dim=0)
        return Variable(q), Variable(p)
