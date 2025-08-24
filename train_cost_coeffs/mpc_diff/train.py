# il_bicycle_exp.py
import os
import time
import argparse
import numpy as np
import torch
from torch import optim
from train_cost_coeffs.mpc_diff.dynamics_bicycle import BicycleDx

from mpc.mpc import mpc
from mpc.mpc.mpc import QuadCost


def make_constant_cost(q_vec, p_vec, T, n_batch, device):
    """
    Build (Q, p) tensors for QuadCost:
      Q: T x B x (nx+nu) x (nx+nu)  (diagonal only, but stored as full diag matrices)
      p: T x B x (nx+nu)
    """
    n_sc = q_vec.numel()
    Q = torch.diag(q_vec).unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1).to(device)
    p = p_vec.unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1).to(device)
    return Q, p


def sample_path(T, kind="straight", amplitude=3.0, frequency=0.2, device="cpu"):
    # (T+1, 2) path points [x, y]
    x = torch.linspace(0, 30, T + 1, device=device)
    if kind == "straight":
        y = torch.zeros_like(x)
    else:
        y = amplitude * torch.sin(frequency * x)
    return torch.stack((x, y), dim=1)


def sample_initial_state_near_path(
    path, max_pos_offset=2.0, max_heading_offset=0.7, max_speed=6.0
):
    # x0 = [X, Y, cos(th), sin(th), v]
    T1 = path.shape[0]
    i = torch.randint(0, T1 - 1, (1,)).item()
    p = path[i]
    p2 = path[i + 1]
    tangent = p2 - p
    theta = (
        torch.atan2(tangent[1], tangent[0])
        + (torch.rand(1) - 0.5) * 2 * max_heading_offset
    )
    v = torch.rand(1) * max_speed
    pos_noise = (torch.rand(2) - 0.5) * 2 * max_pos_offset
    X = p[0] + pos_noise[0]
    Y = p[1] + pos_noise[1]
    x0 = torch.tensor(
        [
            X.item(),
            Y.item(),
            torch.cos(theta).item(),
            torch.sin(theta).item(),
            v.item(),
        ],
        dtype=torch.float32,
    )
    return x0


def run_mpc(
    dx, x_init, q_vec, p_vec, T, u_init=None, u_lower=None, u_upper=None, n_batch=1
):
    device = x_init.device
    nx, nu = dx.n_state, dx.n_ctrl
    Q, p = make_constant_cost(q_vec, p_vec, T, n_batch, device)

    ctrl = mpc.MPC(
        nx,
        nu,
        T,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=100,
        n_batch=n_batch,
        exit_unconverged=False,
        verbose=0,
        u_init=u_init,  # warm-start actions: T x B x nu
        grad_method=mpc.GradMethods.AUTO_DIFF,
    )

    x_init_b = x_init.unsqueeze(0) if x_init.ndim == 1 else x_init  # B x nx
    x_traj, u_traj, _ = ctrl(x_init_b, QuadCost(Q, p), dx)
    return x_traj, u_traj  # shapes: T x B x nx, T x B x nu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_batch", type=int, default=32)
    parser.add_argument("--n_epoch", type=int, default=300)
    parser.add_argument("--learn_rate", type=float, default=0.006)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--path_kind", type=str, default="straight", choices=["straight", "sine"]
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"

    # --- Dynamics
    dx = BicycleDx().to(device)
    nx, nu = dx.n_state, dx.n_ctrl
    n_sc = nx + nu

    # --- "Expert" cost (fixed ground truth)
    true_q, true_p = dx.get_true_obj()
    true_q = true_q.to(device)
    true_p = true_p.to(device)

    # --- Learnable cost (diagonal only)
    # Parameterize diag via logits -> sigmoid -> positive, and p via p = sqrt(q) * learn_p (as in their repo)
    learn_q_logit = torch.zeros_like(true_q, device=device, requires_grad=True)
    learn_p_raw = torch.zeros_like(true_p, device=device, requires_grad=True)

    opt = optim.RMSprop([learn_q_logit, learn_p_raw], lr=args.learn_rate)

    # Warm-start action buffer for MPC (helps stability)
    warmstart = torch.zeros(args.T, args.n_batch, nu, device=device)

    # Training loop
    for epoch in range(1, args.n_epoch + 1):
        # --- sample a mini-batch of initial states and paths
        path = sample_path(args.T, kind=args.path_kind, device=device)
        x_inits = torch.stack(
            [
                sample_initial_state_near_path(path).to(device)
                for _ in range(args.n_batch)
            ],
            dim=0,
        )

        # --- Expert rollouts (targets)
        with torch.no_grad():
            expert_x, expert_u = run_mpc(
                dx,
                x_inits,
                true_q,
                true_p,
                args.T,
                u_init=warmstart,
                n_batch=args.n_batch,
            )

        # --- Predicted rollouts under learned cost
        q_hat = torch.sigmoid(learn_q_logit)  # (n_sc,)
        p_hat = torch.sqrt(q_hat) * learn_p_raw  # (n_sc,)
        pred_x, pred_u = run_mpc(
            dx, x_inits, q_hat, p_hat, args.T, u_init=warmstart, n_batch=args.n_batch
        )

        # update warmstart with current pred actions (T x B x nu)
        warmstart = pred_u.detach()

        # --- Imitation loss on actions (+ small state loss helps sometimes)
        im_u = (expert_u.detach() - pred_u).pow(2).mean()
        im_x = (expert_x.detach() - pred_x).pow(2).mean()
        loss = im_u + 0.1 * im_x

        opt.zero_grad()
        loss.backward()
        # OPTIONAL: freeze some entries if you only want to learn specific diagonals
        # e.g., keep X/cos(theta) costs = 0:
        # idx_freeze = torch.tensor([0, 2], device=device)  # indices in state diag
        # learn_q_logit.grad[idx_freeze] = 0.0
        opt.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                q_err = torch.norm(q_hat - true_q).item()
                p_err = torch.norm(p_hat - true_p).item()
                print(
                    f"[{epoch:04d}] loss={loss.item():.4f} | ||q-q*||={q_err:.4f} ||p-p*||={p_err:.4f}"
                )

    print("\n=== Learned cost (diag only) ===")
    with torch.no_grad():
        q_learn = torch.sigmoid(learn_q_logit).cpu()
        p_learn = torch.sqrt(q_learn) * learn_p_raw.cpu()
        print("q* (true):   ", true_q.cpu().numpy())
        print("q  (learned):", q_learn.numpy())
        print("p* (true):   ", true_p.cpu().numpy())
        print("p  (learned):", p_learn.numpy())


if __name__ == "__main__":
    main()
