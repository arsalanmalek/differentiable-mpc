"""
train.py

Imitation learning to learn cost co-efficients.
Learns structured goal_weights (state-only).
"""

# TODO: dynamics rollout visualization for true and each epoch prediction on test set

import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dynamics import BicycleDynamics
from objective import QuadraticCost, ExpertQuadraticCost

from mpc.mpc import mpc
from mpc.mpc.mpc import QuadCost


def get_random_state():
    """
    Get random state within a certain bounds
    state: [X coord, Y coord, cos-theta, sin-theta, speed]
    """
    x = (torch.rand(1).item() - 0.5) * 10.0  # X coord: -5 to 5
    y = (torch.rand(1).item() - 0.5) * 6.0  # Y coord: -3 to 3
    th = (torch.rand(1).item() - 0.5) * 2.0  # theta: -1 to 1
    v = torch.rand(1).item() * 10.0  # speed: 0 to 10
    return torch.tensor([x, y, np.cos(th), np.sin(th), v], dtype=torch.float32)


def repeat_Qp(
    q_vec,
    p_vec: torch.Tensor,
    n_batch: int,
    count: int,
):
    """
    Creates Q matrix out of q vec
    And repeat Q and p `count` times

    Returns:
    - Q (count x B x n_sc x n_sc ) and p (count x B x n_sc)
    """
    # n_sc = q_vec.numel()
    Q_mat = torch.diag(q_vec)
    Q = Q_mat.unsqueeze(0).unsqueeze(0).repeat(count, n_batch, 1, 1)
    p = p_vec.unsqueeze(0).unsqueeze(0).repeat(count, n_batch, 1)
    return Q, p


def get_expert_objective(expertCost: ExpertQuadraticCost, horizon: int, device: str):
    """
    Returns Q and p tiled for horizon, where Q and p are constructed using target
     state|ctrl and target coeffs
    """
    q_true, p_true = expertCost.get_true_objective()
    q_true = q_true
    p_true = p_true
    return repeat_Qp(q_true.to(device), p_true.to(device), batch=1, count=horizon)


# -------------------------
# data generation (expert)
# -------------------------
def get_expert_trajectories(
    dx, Q_T, p_T, T, n_rollouts, device, u_lower=None, u_upper=None
):
    """
    Generate n_rollouts expert trajectories using MPC with expert cost.
    Returns x_rollouts (N, T, nx), u_rollouts (N, T, nu), q_true_arr, p_true_arr
    """
    nx, nu = dx.n_state, dx.n_ctrl

    ctrl = mpc.MPC(
        nx,
        nu,
        T,
        u_lower=u_lower,
        u_upper=u_upper,
        lqr_iter=200,
        verbose=0,
        exit_unconverged=False,
        n_batch=1,
        grad_method=mpc.GradMethods.AUTO_DIFF,
    )

    x_rollouts = []
    u_rollouts = []
    for i in range(n_rollouts):
        rand_x0 = get_random_state().to(device)
        print(i)
        x_traj, u_traj, _ = ctrl(rand_x0.unsqueeze(0), QuadCost(Q_T, p_T), dx)
        x_traj = x_traj.squeeze(1).detach().cpu().numpy()  # (T, nx)
        u_traj = u_traj.squeeze(1).detach().cpu().numpy()  # (T, nu)

        x_rollouts.append(x_traj)
        u_rollouts.append(u_traj)

    x_rollouts = np.stack(x_rollouts, axis=0).astype(np.float32)
    u_rollouts = np.stack(u_rollouts, axis=0).astype(np.float32)
    return x_rollouts, u_rollouts


# -------------------------
# IL Exp (training loop)
# -------------------------
def train(
    T=10,
    n_batch=12,
    n_epochs=50,
    n_train=128,
    n_test=12,
    learn_rate=5e-3,
    freeze_indices=[],
    device="cpu",
):
    torch.manual_seed(0)
    np.random.seed(0)

    ctrl_coeff_val = 0.001
    state_coeff_val_init = 0.1

    # target_state order: [X, Y, cosθ, sinθ, goal_speed]
    target_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.5], dtype=torch.float32)
    target_state_coeffs = torch.tensor([0.0, 0.15, 0.0, 1.0, 0.5], dtype=torch.float32)

    dx = BicycleDynamics().to(device)
    nx, nu = dx.n_state, dx.n_ctrl

    expertCost = ExpertQuadraticCost(
        nx, nu, target_state, target_state_coeffs, ctrl_coeff_val, device
    )

    learnableCost = QuadraticCost(
        nx, nu, target_state, state_coeff_val_init, ctrl_coeff_val, device
    )

    ### EXPERT DATASET
    # Generate true objective q and p vectors (dependent on target state|ctrl and coeffs)
    Q_T_expert, p_T_expert = get_expert_objective(expertCost, T, device)

    print("Create expert trajectories train...")
    X_train, U_train = get_expert_trajectories(
        dx, Q_T_expert, p_T_expert, T, n_train, device
    )
    print("Create expert trajectories test...")
    X_test, U_test = get_expert_trajectories(
        dx, Q_T_expert, p_T_expert, T, n_test, device
    )

    ### CREATE TRAIN DATASET

    x0_train = torch.tensor(
        X_train[:, 0, :], dtype=torch.float32, device=device
    )  # (N, nx)
    u_train = torch.tensor(U_train, dtype=torch.float32, device=device)  # (N, T, nu)

    x0_test = torch.tensor(X_test[:, 0, :], dtype=torch.float32, device=device)
    u_test = torch.tensor(U_test, dtype=torch.float32, device=device)

    train_ds = TensorDataset(x0_train, u_train)
    test_ds = TensorDataset(x0_test, u_test)

    train_loader = DataLoader(train_ds, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=n_batch, shuffle=False)

    # freeze mask for state indices (length nx). Indices refer to state entries only (0..nx-1)
    freeze_mask_state = torch.zeros(nx, dtype=torch.bool, device=device)
    for idx in freeze_indices:
        if 0 <= idx < nx:
            freeze_mask_state[idx] = True

    # optimizer = torch.optim.Adam(learnableCost.parameters(), lr=learn_rate)
    optimizer = torch.optim.RMSprop(learnableCost.parameters(), lr=learn_rate)

    # helper: run MPC given q_vec, p_vec and inputs (batched)
    def run_mpc_batch(x0_batch, q_vec, p_vec):
        B = x0_batch.shape[0]
        Q_T, p_T = repeat_Qp(q_vec, p_vec, B, T)
        ctrl = mpc.MPC(
            nx,
            nu,
            T,
            u_lower=None,
            u_upper=None,
            lqr_iter=300,
            verbose=0,
            exit_unconverged=False,
            n_batch=B,
            grad_method=mpc.GradMethods.AUTO_DIFF,
        )
        x_traj, u_traj, _ = ctrl(x0_batch, QuadCost(Q_T, p_T), dx)
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
        counter = 0
        # training
        for x0_batch, u_batch in train_loader:
            counter += 1
            B = x0_batch.shape[0]

            # build learner q,p from learn_cost
            q_vec, p_vec = learnableCost()

            # run MPC (batch)
            print("batch: ", counter)
            x_pred, u_pred = run_mpc_batch(x0_batch, q_vec, p_vec)

            u_pred_bt = u_pred.transpose(0, 1)  # B x T x nu

            # imitation loss (action MSE)
            im_loss = (u_batch.to(device) - u_pred_bt).pow(2).mean()
            optimizer.zero_grad()
            im_loss.backward()

            # enforce freeze: zero gradients for frozen state indices in raw_goal_weights
            print(learnableCost.raw_goal_weights.grad.data)
            if freeze_mask_state.any():
                if learnableCost.raw_goal_weights.grad is not None:
                    learnableCost.raw_goal_weights.grad.data[freeze_mask_state] = 0.0
            optimizer.step()

            epoch_loss += im_loss.item() * B
            n_seen += B

        epoch_loss /= float(n_seen)
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            with torch.no_grad():
                _, best_p_vec, best_q_state = learnableCost()
        print(
            "\n\nEpoch Loss: ", epoch_loss, "\nWeights: ", learnableCost.goal_weights()
        )
        print("\nFinal states: ", x_pred[:, -1], "\n\n")

    # final test evaluation
    test_loss = 0.0
    n_seen = 0
    # with torch.no_grad():
    for x0_batch, u_batch in test_loader:
        q_vec, p_vec = learnableCost()
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
    if best_p_vec is not None:
        print("Best learned p_state vec:", np.round(best_p_vec.cpu().numpy(), 6))
