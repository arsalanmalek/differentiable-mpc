import torch


class BicycleDynamics(torch.nn.Module):
    """
    Defines non linear bicycle dynamics model
    based on the examples from mpc repo

    `forward` returns the next state of the dynamics model\n
    - `State`:   [X, Y, cos(theta), sin(theta), v]   (n_state = 5)\n
    - `Control`: [a, delta]                          (n_ctrl  = 2)

    -----------

    The dynamics are same here from `simple_mpc` except that for theta,
    we have 2 different components so its always continuous and stable
    """

    params: torch.Tensor

    def __init__(self, params=None):
        super().__init__()
        # counts
        self.n_state = 5  # (X, Y, cos, sin, v)
        self.n_ctrl = 2  # (a, delta)
        # control limits
        self.accel_lim = 2.0
        self.steer_lim = 0.5

        if params is None:
            params = torch.tensor(
                (0.1, 1.0),  # dt, L - wheel base
                dtype=torch.float32,
            )

        self.register_buffer("params", params)  # part of state_dict, wo being learnable

    def forward(self, state: torch.Tensor, u: torch.Tensor):
        """
        Roll out non-linear dynamics, step size = dt
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
