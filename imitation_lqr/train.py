#!/usr/bin/env python3

import torch
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

import numpy as np
import numpy.random as npr


import sys

import time
import os
import shutil
import pickle as pkl
import collections

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from mpc.mpc import mpc
from mpc.mpc.mpc import GradMethods, QuadCost, LinDx
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Verbose", call_pdb=1)
# sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

import argparse
import setproctitle

# import setGPU


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
    setproctitle.setproctitle("bamos.lqr." + t + ".{}".format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work, t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = "cuda" if args.cuda else "cpu"

    n_state, n_ctrl = args.n_state, args.n_ctrl
    n_sc = n_state + n_ctrl

    expert_seed = 42
    assert expert_seed != args.seed
    torch.manual_seed(expert_seed)

    # Q == Ct [where C= [[Q' 0][0 R]]]
    # p == ct
    Q = torch.eye(n_sc)
    p = torch.randn(n_sc)  # normalized: probably m=0, v=1

    alpha = 0.2

    expert = dict(
        Q=torch.eye(n_sc).to(device),
        p=torch.randn(n_sc).to(device),
        A=(torch.eye(n_state) + alpha * torch.randn(n_state, n_state)).to(
            device
        ),  # This initialization ensures that A is stable
        B=torch.randn(n_state, n_ctrl).to(device),
    )
    fname = os.path.join(args.save, "expert.pkl")
    with open(fname, "wb") as f:
        pkl.dump(expert, f)

    torch.manual_seed(args.seed)
    # F = [A|B]
    # xt+1 = A.xt + B.ut --> [A|B]*[xt ut].T
    # xt+1 = F @ tau_t
    # no f_t term here
    A = (
        (torch.eye(n_state) + alpha * torch.randn(n_state, n_state))
        .to(device)
        .requires_grad_()
    )
    B = torch.randn(n_state, n_ctrl).to(device).requires_grad_()

    # u_lower, u_upper = -10., 10.
    u_lower, u_upper = None, None
    delta = u_init = None

    fname = os.path.join(args.save, "losses.csv")
    loss_f = open(fname, "w")
    loss_f.write("im_loss,mse\n")
    loss_f.flush()

    def get_loss(x_init, _A, _B):
        F = (
            torch.cat(
                (expert["A"], expert["B"]), dim=1
            )  # n_state x n_state, n_state x n_ctrl --> n_state x (n_state + n_ctrl)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(args.T, n_batch, 1, 1)
        )  # [T= Horizon, B=batch_count, n_state, n_state + n_ctrl]
        # Getting expert's MPC trajectory (x, u)s using random initial states
        x_true, u_true, objs_true = mpc.MPC(
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
        )(x_init, QuadCost(expert["Q"], expert["p"]), LinDx(F))

        F = (
            torch.cat((_A, _B), dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(args.T, n_batch, 1, 1)
        )
        x_pred, u_pred, objs_pred = mpc.MPC(
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
        )(x_init, QuadCost(expert["Q"], expert["p"]), LinDx(F))

        traj_loss = torch.mean((u_true - u_pred) ** 2) + torch.mean(
            (x_true - x_pred) ** 2
        )  # MSE loss of complete trajectory's actions and states in expert and learner
        return traj_loss

    # here A and B are the parameters to be optimized
    opt = optim.RMSprop(
        (A, B), lr=1e-2
    )  # could try Adam (which has momentum and might not be suitable for non-stationary objectives like in RL;
    # still works better in some cases)

    n_batch = 128
    for i in range(5000):  # 128*5k samples to guess the dynamics model
        x_init = torch.randn(n_batch, n_state).to(device)  # randomized initial states
        traj_loss = get_loss(x_init, A, B)

        opt.zero_grad()
        traj_loss.backward()
        opt.step()

        model_loss = torch.mean((A - expert["A"]) ** 2) + torch.mean(
            (B - expert["B"]) ** 2
        )  # Distance between dynamics of expert and predicted; this is not the loss which is optimized

        loss_f.write("{},{}\n".format(traj_loss.item(), model_loss.item()))
        loss_f.flush()

        plot_interval = 100
        if i % plot_interval == 0:
            os.system('./plot.py "{}" &'.format(args.save))
            print(A, expert["A"])
        print(
            "{:04d}: traj_loss: {:.4f} model_loss: {:.4f}".format(
                i, traj_loss.item(), model_loss.item()
            )
        )

        # except KeyboardInterrupt: TODO
        #     raise
        # except Exception as e:
        #     # print(e)
        #     # pass
        #     raise


if __name__ == "__main__":
    main()
