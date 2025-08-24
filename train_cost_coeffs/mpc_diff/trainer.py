"""
IL_Exp_bicycle.py

Imitation learning experiment for bicycle dynamics.
Learns structured goal_weights (state-only). p is derived as
p_state = - sqrt(q_state) * goal_state, so cost = (x - x_goal)^T Q (x - x_goal) + u^T R u.

Default: freeze state indices [0, 3] (X pos and sin(theta)).
"""

import os
from os.path import join, dirname
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(dirname(dirname(dirname(__file__)))))

from mpc.mpc import mpc
from mpc.mpc.mpc import QuadCost
from train_cost_coeffs.mpc_diff.dynamics_bicycle import BicycleDynamicsWithCost


# -------------------------
# Learnable structured cost module
# -------------------------
class LearnableGoalCost(nn.Module):
    """
    Learns state goal weights (n_state) via an unconstrained parameter raw_goal_weights.
    goal_weights = softplus(raw_goal_weights) (=> positive).
    p_state is derived as: p_state = - sqrt(q_state) * goal_state
    Control penalties q_ctrl are kept fixed (scalar from dx.ctrl_penalty).
    """

    def __init__(
        self,
        n_state,
        n_ctrl,
        goal_state,
        init_val=0.1,
        ctrl_penalty=0.001,
        device="cpu",
    ):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.ctrl_penalty = float(ctrl_penalty)

        # raw param initialised so softplus(raw) ~ init_val
        # inverse softplus approx: raw ≈ log(exp(init)-1); but init small -> raw ≈ log(init)
        init_tensor = torch.full(
            (n_state,), init_val, dtype=torch.float32, device=device
        )
        # to get raw such that softplus(raw) ~= init_val, we use inverse via log(exp(x)-1)
        # avoid numerical issues for very small init_val:
        raw_init = torch.log(torch.expm1(init_tensor.clamp(min=1e-6)))
        self.raw_goal_weights = nn.Parameter(raw_init)

        # store goal_state as buffer (not learned)
        self.register_buffer("goal_state", goal_state.clone().detach().to(device))

    def goal_weights(self):
        # softplus ensures positive weights
        return F.softplus(self.raw_goal_weights) + 1e-8

    def forward(self):
        """
        Returns (q_vec, p_vec, q_state) where:
         - q_vec shape: (n_state + n_ctrl,)
         - p_vec shape: (n_state + n_ctrl,)
         - q_state shape: (n_state,)
        """
        q_state = self.goal_weights()  # (n_state,)
        q_ctrl = self.ctrl_penalty * torch.ones(
            self.n_ctrl, dtype=torch.float32, device=q_state.device
        )
        q_vec = torch.cat((q_state, q_ctrl), dim=0)

        sqrt_q_state = torch.sqrt(torch.clamp(q_state, min=1e-12))
        p_state = -sqrt_q_state * self.goal_state
        p_ctrl = torch.zeros(self.n_ctrl, dtype=torch.float32, device=q_state.device)
        p_vec = torch.cat((p_state, p_ctrl), dim=0)

        return q_vec, p_vec, q_state


# -------------------------
# helper: build QuadCost tensors (T x B x n_sc x n_sc ) and p (T x B x n_sc)
# -------------------------
def make_Qp_for_horizon(q_vec, p_vec, T, n_batch, device):
    n_sc = q_vec.numel()
    Q_mat = torch.diag(q_vec).to(device)
    Q = Q_mat.unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
    p = p_vec.to(device).unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1)
    return Q, p


# -------------------------
# data generation (expert)
# -------------------------
def generate_expert_dataset(dx, T, n_rollouts, device, u_lower=None, u_upper=None):
    """
    Generate n_rollouts expert trajectories using MPC with expert cost.
    Returns x_rollouts (N, T, nx), u_rollouts (N, T, nu), q_true_arr, p_true_arr
    """
    nx, nu = dx.n_state, dx.n_ctrl

    q_true, p_true = dx.get_true_obj()
    q_true = q_true.to(device)
    p_true = p_true.to(device)

    Q_tens, p_tens = make_Qp_for_horizon(q_true, p_true, T, 1, device)

    ctrl = mpc.MPC(
        nx,
        nu,
        T,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        n_batch=1,
        grad_method=mpc.GradMethods.AUTO_DIFF,
    )

    x_rollouts = []
    u_rollouts = []
    for i in range(n_rollouts):
        X0 = (torch.rand(1).item() - 0.5) * 10.0
        Y0 = (torch.rand(1).item() - 0.5) * 4.0
        th0 = (torch.rand(1).item() - 0.5) * 2.0
        v0 = torch.rand(1).item() * 3.0
        x0 = torch.tensor(
            [X0, Y0, np.cos(th0), np.sin(th0), v0], dtype=torch.float32, device=device
        )
        x0_b = x0.unsqueeze(0)

        x_traj, u_traj, _ = ctrl(x0_b, QuadCost(Q_tens, p_tens), dx)
        x_traj = x_traj.squeeze(1).detach().cpu().numpy()  # (T, nx)
        u_traj = u_traj.squeeze(1).detach().cpu().numpy()  # (T, nu)

        x_rollouts.append(x_traj)
        u_rollouts.append(u_traj)

    x_rollouts = np.stack(x_rollouts, axis=0).astype(np.float32)
    u_rollouts = np.stack(u_rollouts, axis=0).astype(np.float32)
    return x_rollouts, u_rollouts, q_true.cpu().numpy(), p_true.cpu().numpy()


# -------------------------
# IL Experiment (training loop)
# -------------------------
def train_il_bicycle(
    device="cpu",
    T=20,
    n_batch=32,
    n_epochs=300,
    n_train=1024,
    n_val=256,
    n_test=256,
    learn_rate=1e-2,
    freeze_indices=None,
):
    torch.manual_seed(0)
    np.random.seed(0)

    dx = BicycleDx().to(device)
    nx, nu = dx.n_state, dx.n_ctrl
    n_sc = nx + nu

    # -------------------------
    # Generate datasets (expert)
    # -------------------------
    print("Generating expert dataset (this may take a bit)...")
    X_train, U_train, q_true_arr, p_true_arr = generate_expert_dataset(
        dx, T, n_train, device
    )
    X_val, U_val, _, _ = generate_expert_dataset(dx, T, n_val, device)
    X_test, U_test, _, _ = generate_expert_dataset(dx, T, n_test, device)

    x0_train = torch.tensor(
        X_train[:, 0, :], dtype=torch.float32, device=device
    )  # (N, nx)
    u_train = torch.tensor(U_train, dtype=torch.float32, device=device)  # (N, T, nu)

    x0_val = torch.tensor(X_val[:, 0, :], dtype=torch.float32, device=device)
    u_val = torch.tensor(U_val, dtype=torch.float32, device=device)

    x0_test = torch.tensor(X_test[:, 0, :], dtype=torch.float32, device=device)
    u_test = torch.tensor(U_test, dtype=torch.float32, device=device)

    train_ds = TensorDataset(x0_train, u_train)
    val_ds = TensorDataset(x0_val, u_val)
    test_ds = TensorDataset(x0_test, u_test)

    train_loader = DataLoader(train_ds, batch_size=n_batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=n_batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=n_batch, shuffle=False)

    # Expert q,p (vectors) (for diagnostic/printing)
    q_true = torch.tensor(q_true_arr, dtype=torch.float32, device=device)
    p_true = torch.tensor(p_true_arr, dtype=torch.float32, device=device)

    # -------------------------
    # Learner: structured LearnableGoalCost (state-only)
    # -------------------------
    learn_cost = LearnableGoalCost(
        n_state=nx,
        n_ctrl=nu,
        goal_state=dx.goal_state,
        init_val=0.1,
        ctrl_penalty=dx.ctrl_penalty,
        device=device,
    )
    learn_cost = learn_cost.to(device)

    # freeze mask for state indices (length nx). Indices refer to state entries only (0..nx-1)
    freeze_mask_state = torch.zeros(nx, dtype=torch.bool, device=device)
    if freeze_indices is not None:
        for idx in freeze_indices:
            if 0 <= idx < nx:
                freeze_mask_state[idx] = True

    optimizer = optim.RMSprop(learn_cost.parameters(), lr=learn_rate)

    # warmstart buffer for MPC (T x B x nu)
    warmstart = torch.zeros(T, n_batch, nu, device=device)

    # helper: run MPC given q_vec, p_vec and inputs (batched)
    def run_mpc_batch(x0_batch, q_vec, p_vec):
        B = x0_batch.shape[0]
        Q_tens, p_tens = make_Qp_for_horizon(q_vec, p_vec, T, B, device)
        ctrl = mpc.MPC(
            nx,
            nu,
            T,
            u_lower=None,
            u_upper=None,
            lqr_iter=50,
            verbose=0,
            exit_unconverged=False,
            n_batch=B,
            grad_method=mpc.GradMethods.AUTO_DIFF,
        )
        x_traj, u_traj, _ = ctrl(x0_batch, QuadCost(Q_tens, p_tens), dx)
        return x_traj, u_traj

    # -------------------------
    # Training loop
    # -------------------------
    best_val_loss = float("inf")
    best_q_state = None
    best_p_vec = None

    for epoch in range(1, n_epochs + 1):
        st = time.time()
        epoch_loss = 0.0
        n_seen = 0

        # training
        for x0_batch, u_batch in train_loader:
            B = x0_batch.shape[0]

            # build learner q,p from learn_cost
            q_vec, p_vec, q_state_vec = learn_cost()

            # run MPC (batch)
            x_pred, u_pred = run_mpc_batch(x0_batch, q_vec, p_vec)

            u_pred_bt = u_pred.transpose(0, 1)  # B x T x nu

            # imitation loss (action MSE)
            im_loss = (u_batch.to(device) - u_pred_bt).pow(2).mean()

            optimizer.zero_grad()
            im_loss.backward()

            # enforce freeze: zero gradients for frozen state indices in raw_goal_weights
            if freeze_mask_state.any():
                if learn_cost.raw_goal_weights.grad is not None:
                    learn_cost.raw_goal_weights.grad.data[freeze_mask_state] = 0.0

            optimizer.step()

            epoch_loss += im_loss.item() * B
            n_seen += B

            # update warmstart with current predictions
            warmstart[:, :B, :] = u_pred.detach()

        epoch_loss /= float(n_seen)

        # validation
        val_loss = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for x0_batch, u_batch in val_loader:
                q_vec, p_vec, _ = learn_cost()
                x_pred, u_pred = run_mpc_batch(x0_batch, q_vec, p_vec)
                u_pred_bt = u_pred.transpose(0, 1)
                val_loss += (u_batch.to(device) - u_pred_bt).pow(
                    2
                ).mean().item() * x0_batch.shape[0]
                n_val_seen += x0_batch.shape[0]
            val_loss /= n_val_seen

        elapsed = time.time() - st
        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with torch.no_grad():
                _, best_p_vec, best_q_state = learn_cost()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                q_state_now = learn_cost.goal_weights().cpu().numpy()
                q_now_full = np.concatenate(
                    [q_state_now, np.array([dx.ctrl_penalty, dx.ctrl_penalty])]
                )
                print(
                    f"[{epoch:04d}] train_loss={epoch_loss:.6f} val_loss={val_loss:.6f} time={elapsed:.2f}s"
                )
                print("  q_state (learned diag):", np.round(q_state_now, 6))
                print("  full q diag (state+ctrl):", np.round(q_now_full, 6))

    # final test evaluation
    test_loss = 0.0
    n_seen = 0
    with torch.no_grad():
        for x0_batch, u_batch in test_loader:
            q_vec, p_vec, _ = learn_cost()
            x_pred, u_pred = run_mpc_batch(x0_batch, q_vec, p_vec)
            u_pred_bt = u_pred.transpose(0, 1)
            test_loss += (u_batch.to(device) - u_pred_bt).pow(
                2
            ).mean().item() * x0_batch.shape[0]
            n_seen += x0_batch.shape[0]
        test_loss /= n_seen

    print("Training done.")
    print("Best val loss:", best_val_loss)
    print("Test loss:", test_loss)
    if best_q_state is not None:
        print("Best learned q_state diag:", np.round(best_q_state.cpu().numpy(), 6))
    return learn_cost


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n_batch", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=256)
    parser.add_argument("--learn_rate", type=float, default=1e-2)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--freeze_idx",
        type=int,
        nargs="*",
        default=[0, 3],
        help="state indices of goal_weights to freeze (0..n_state-1). default [0,3].",
    )
    args = parser.parse_args()

    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    train_il_bicycle(
        device=device,
        T=args.T,
        n_batch=args.n_batch,
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        learn_rate=args.learn_rate,
        freeze_indices=args.freeze_idx,
    )
