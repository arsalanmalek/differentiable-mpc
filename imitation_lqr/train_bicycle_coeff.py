import torch
import torch.optim as optim

import sys

import os
import shutil

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from mpc.mpc import mpc
from bicycle import BicycleDx
from mpc.mpc.mpc import GradMethods, QuadCost, LinDx
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Verbose", call_pdb=1)

import argparse
import setproctitle


def expert_mpc(state):
    expert_seed = 42
    torch.manual_seed(expert_seed)


def compute_objective_quadratic(
    path_points, goal_speed, wp=0.15, ws=0.5, wa=1.0, n_state=4, n_ctrl=2, device="cpu"
):
    """
    Create per-timestep quadratic cost (C, c) from path and goal speed.

    States: [x, y, theta, v]
    Controls: [u1, u2] (not penalized here unless added manually)

    Returns:
        C (n_state+n_ctrl, n_state+n_ctrl)
        c (n_state+n_ctrl)
    """

    n_sc = n_state + n_ctrl
    C = torch.zeros(n_sc, n_sc, device=device)
    c = torch.zeros(n_sc, device=device)

    # Reference point: here we just take the first path point
    x_ref, y_ref = path_points[0, 0].item(), path_points[0, 1].item()

    # Heading reference: tangent to first segment
    if path_points.shape[0] >= 2:
        dx = path_points[1, 0] - path_points[0, 0]
        dy = path_points[1, 1] - path_points[0, 1]
        theta_ref = torch.atan2(dy, dx).item()
    else:
        theta_ref = 0.0

    # --- Diagonal quadratic terms (trainable) ---
    coeffs = torch.tensor([wp, wp, wa, ws], device=device, requires_grad=True)
    C[range(n_state), range(n_state)] = coeffs

    # --- Linear terms ---
    c[0] = -2 * wp * x_ref
    c[1] = -2 * wp * y_ref
    c[2] = -2 * wa * theta_ref
    c[3] = -2 * ws * goal_speed

    return C, c, coeffs


def sample_initial_state_near_path(
    path, max_pos_offset=5.0, max_heading_offset=1.0, max_speed=10.0
):
    """
    Sample a random initial state near a given path.

    Args:
        path: Tensor of shape (T+1, 2), path coordinates
        max_pos_offset: max x/y offset from path point (meters)
        max_heading_offset: max deviation from path tangent (radians)
        max_speed: upper bound for random initial speed (m/s)

    Returns:
        x0: Tensor of shape (n_state,) = [x, y, θ, v]
    """
    T_plus_1 = path.shape[0]
    idx = torch.randint(0, T_plus_1 - 1, (1,)).item()

    # Path point and next for tangent direction
    pt = path[idx]
    pt_next = path[idx + 1]
    tangent = pt_next - pt
    base_theta = torch.atan2(tangent[1], tangent[0])

    # Add noise
    pos_noise = torch.randn(2) * max_pos_offset
    theta_noise = (torch.rand(1) - 0.5) * 2 * max_heading_offset
    v = torch.rand(1) * max_speed

    x = pt[0] + pos_noise[0]
    y = pt[1] + pos_noise[1]
    theta = base_theta + theta_noise

    x0 = torch.tensor([x, y, theta.item(), v.item()])
    return x0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state", type=int, default=3)  # x - state vector elements
    parser.add_argument("--n_ctrl", type=int, default=3)  # u - control vector elements
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--save", type=str)
    parser.add_argument("--work", type=str, default="work")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = ".".join(
        ["{}={}".format(x, getattr(args, x)) for x in ["n_state", "n_ctrl", "T"]]
    )
    setproctitle.setproctitle("lqr." + t + ".{}".format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work, t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = "cuda" if args.cuda else "cpu"

    n_state, n_ctrl = args.n_state, args.n_ctrl
    n_sc = n_state + n_ctrl
    # alpha = 0.2

    expert_C, expert_c = compute_objective_quadratic(
        path,
        goal_speed=10.0,
        wp=0.15,
        ws=0.5,
        wa=1.0,
        n_state=n_state,
        n_ctrl=n_ctrl,
        device=device,
    )
    torch.manual_seed(args.seed)
    # F = [A|B]
    # xt+1 = A.xt + B.ut --> [A|B]*[xt ut].T
    # xt+1 = F @ tau_t
    # no f_t term here
    # A = (torch.eye(n_state) + alpha * torch.randn(n_state, n_state)).to(device)
    # B = torch.randn(n_state, n_ctrl).to(device)
    # OBJ = tau.C.tau + c
    # C = torch.eye(n_sc).to(device).requires_grad_()
    # c = torch.randn(n_sc).to(device).requires_grad_()

    C, c, cost_coeffs = compute_objective_quadratic(
        path,
        goal_speed=10.0,
        wp=0.1,
        ws=0.1,
        wa=0.1,
        n_state=n_state,
        n_ctrl=n_ctrl,
        device=device,
    )
    opt = torch.optim.RMSprop([cost_coeffs], lr=1e-3)

    # u_lower, u_upper = -10., 10.
    u_lower, u_upper = None, None
    delta = u_init = None

    fname = os.path.join(args.save, "losses.csv")
    loss_f = open(fname, "w")
    loss_f.write("im_loss,mse\n")
    loss_f.flush()

    def get_loss(x_init, path):
        x_true, u_true, __ = mpc.MPC(
            n_state,
            n_ctrl,
            args.T,
            u_lower=u_lower,
            u_upper=u_upper,
            u_init=u_init,
            lqr_iter=100,
            verbose=-1,
            exit_unconverged=False,
            detach_unconverged=False,
            n_batch=n_batch,
        )(x_init, QuadCost(expert_C, expert_c), BicycleDx())

        x_pred, u_pred, __ = mpc.MPC(
            n_state,
            n_ctrl,
            args.T,
            u_lower=u_lower,
            u_upper=u_upper,
            u_init=u_init,
            lqr_iter=100,
            verbose=-1,
            exit_unconverged=False,
            detach_unconverged=False,
            n_batch=n_batch,
        )(x_init, QuadCost(C, c), BicycleDx())

        traj_loss = torch.mean((u_true - u_pred) ** 2) + torch.mean(
            (x_true - x_pred) ** 2
        )
        return traj_loss

    n_batch = 128
    T = args.T

    # 128 * 5k samples to guess the dynamics model
    for i in range(5000):
        path = torch.stack(
            [
                torch.linspace(0, 30, T + 1),
                torch.zeros(T + 1),
                #    + torch.randn(1)
            ],
            dim=1,
        )
        x_init = sample_initial_state_near_path(path)
        traj_loss = get_loss(x_init, path)

        opt.zero_grad()
        traj_loss.backward()
        opt.step()

        model_loss = torch.mean((C - expert_C) ** 2) + torch.mean((c - expert_c) ** 2)

        loss_f.write("{},{}\n".format(traj_loss.item(), model_loss.item()))
        loss_f.flush()

        plot_interval = 100
        if i % plot_interval == 0:
            os.system('./plot.py "{}" &'.format(args.save))
        print(
            "{:04d}: traj_loss: {:.4f} model_loss: {:.4f}".format(
                i, traj_loss.item(), model_loss.item()
            )
        )


if __name__ == "__main__":
    main()
