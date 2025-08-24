import torch
from torch import nn


class BicycleDynamicsWithCost(nn.Module):
    """
    Defines non linear dynamics model and quadratic cost matrices
    based on the examples in mpc repo

    - `forward` returns the next state of the dynamics model
    - `get_true_obj` returns the cost matrix Ct and vector ct based on goal state
    and weights defined in class initialization

    State: [X, Y, cos(theta), sin(theta), v]  (n_state = 5)
    Control: [a, delta]                       (n_ctrl  = 2)
    """

    def __init__(self, params=None):
        super().__init__()
        self.n_state = 5
        self.n_ctrl = 2

        # model parameters
        if params is None:
            # dt - delta time, L - vehicle wheel base
            params = torch.tensor((0.1, 1.0), dtype=torch.float32)

        # register_buffer puts tensor in state_dict
        # without making them learnable
        self.register_buffer("params", params)

        # goal_state order: [X, Y, cosθ, sinθ, goal_speed]
        goal_state = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.5], dtype=torch.float32)
        self.register_buffer("goal_state", goal_state)

        # default diag weights for expert (state)
        # We want to learn these in our example
        goal_weights = torch.tensor([0.0, 0.15, 0.0, 1.0, 0.5], dtype=torch.float32)
        self.register_buffer("goal_weights", goal_weights)
        self.ctrl_penalty = 0.001

        # control limits
        self.accel_lim = 2.0
        self.steer_lim = 0.5

    def forward(self, state, u):
        """
        Roll out non-linear dynamics
        input:
          state: (...,5) with (X, Y, cos, sin, v) OR (batch,5)
          u:     (...,2) with (a, delta)
        returns next state same shape
        """
        squeeze = state.ndim == 1
        if squeeze:
            state = state.unsqueeze(0)
            u = u.unsqueeze(0)

        dt = float(self.params[0].item())
        L = float(self.params[1].item())

        # clamp controls
        a = torch.clamp(u[..., 0], -self.accel_lim, self.accel_lim)
        delta = torch.clamp(u[..., 1], -self.steer_lim, self.steer_lim)

        X, Y, cth, sth, v = torch.unbind(state, dim=-1)
        th = torch.atan2(sth, cth)

        beta = torch.atan(0.5 * torch.tan(delta))  # Lf = Lr = L/2

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

        # renormalize cos/sin to avoid drift
        norm = torch.sqrt(cth * cth + sth * sth + 1e-12)
        cth = cth / norm
        sth = sth / norm

        xnext = torch.stack((X, Y, cth, sth, v), dim=-1)
        if squeeze:
            xnext = xnext.squeeze(0)
        return xnext

    def get_true_obj(self):
        """
        Gives linear quadratic cost based on the defined goal states and their weights

        Returns (q, p) vectors where q is diagonal of Ct and p is ct of cost:
          q: (n_state + n_ctrl,) diag entries
          p: (n_state + n_ctrl,) linear term such that p_state = -sqrt(q_state)*goal_state
        """
        q_state = self.goal_weights.clone().detach()
        q_ctrl = self.ctrl_penalty * torch.ones(self.n_ctrl, dtype=torch.float32)
        q = torch.cat((q_state, q_ctrl), dim=0)
        sqrt_q_state = torch.sqrt(torch.clamp(q_state, min=1e-12))
        p_state = -sqrt_q_state * self.goal_state
        p_ctrl = torch.zeros(self.n_ctrl, dtype=torch.float32)
        p = torch.cat((p_state, p_ctrl), dim=0)
        return q, p
