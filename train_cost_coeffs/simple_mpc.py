import os
import torch
from torch import nn
import torch.optim as optim
from datetime import datetime


def bicycle_model_step(x, u, dt=0.1, L=2.5):
    """
    x: (batch, 4) => [X, Y, θ, v]
    u: (batch, 2) => [a, δ]
    returns: next state
    """
    X, Y, theta, v = x[0], x[1], x[2], x[3]
    a, delta = u[0], u[1]

    beta = torch.atan(0.5 * torch.tan(delta))  # assume Lf = Lr = L/2
    dx = torch.stack(
        [
            v * torch.cos(theta + beta),
            v * torch.sin(theta + beta),
            v / L * torch.sin(beta),
            a,
        ],
    )

    return x + dt * dx


def rollout(x0, u_seq, dt, L):
    x_seq = [x0]
    x = x0
    for u in u_seq:
        x = bicycle_model_step(x, u, dt, L)
        x_seq.append(x)
    return torch.stack(x_seq, dim=0)  # (T+1, 4)


def alignment_objective(x, path):
    pos = x[:2]
    heading = x[2]

    dists = ((path - pos) ** 2).sum(dim=1)
    min_idx = torch.argmin(dists)

    idx_start = max(min_idx - 1, 0)
    idx_end = min(min_idx + 1, len(path) - 1)

    path_segment = path[idx_start : idx_end + 1]
    path_vector = path_segment[-1] - path_segment[0]

    # Normalize vectors
    path_dir = path_vector / (path_vector.norm() + 1e-8)
    vehicle_dir = torch.stack([torch.cos(heading), torch.sin(heading)])

    cos_angle = torch.dot(vehicle_dir, path_dir)
    return 1 - cos_angle**2


# Fix objective function:
def compute_objective(x_seq, path_points, goal_speed, wp=0.03, ws=25.0, wa=0.2):
    pos_error = []
    angle_error = []
    for i in range(x_seq.shape[0]):
        dists = compute_dist(x_seq[i], path_points)
        min_id = torch.argmin(dists)
        pos_error.append(dists[min_id])
        angle_error.append(alignment_objective(x_seq[i], path_points))
    pos_error = torch.stack(pos_error)
    angle_error = torch.stack(angle_error)
    speed_error = (x_seq[:, 3] - goal_speed) ** 2
    cost = (wp * pos_error + ws * speed_error + wa * angle_error).sum() * 3
    final_stage_cost = (
        wp * pos_error[-1] + ws * speed_error[-1] + wa * angle_error[-1]
    ).sum() * 3
    return cost, final_stage_cost


def compute_dist(x, path_points):
    dists = (path_points[:, 1] - x[1]) ** 2
    return dists


def mpc_optimize(
    x0,
    path_points,
    goal_speed,
    T=40,
    dt=0.1,
    L=1,
    steer_lim=0.5,
    accel_lim=2.0,
    lr=0.015,
    iters=200,
    steps=5,
):
    """
    x0: (1, 4)
    path_points: (T+1, 2)
    """

    # TODO: convexity of objective could help in better convergence
    u_seq = nn.Parameter(torch.zeros(T + steps - 1, 2))  # [a, δ]
    optimizer = optim.Adam([u_seq], lr=lr)

    init_cost = None
    c_x = x0
    for start_pos in range(steps):
        # print(f"\nStep: {start_pos}")
        c_u_seq = u_seq[start_pos : T + start_pos]
        for __ in range(iters):
            optimizer.zero_grad()

            u_clipped = torch.cat(
                [
                    torch.clamp(c_u_seq[:, :1], -accel_lim, accel_lim),
                    torch.clamp(c_u_seq[:, 1:], -steer_lim, steer_lim),
                ],
                dim=1,
            )

            x_seq = rollout(c_x, u_clipped, dt, L)
            cost, final_stage_cost = compute_objective(x_seq, path_points, goal_speed)
            print(f"Cost: {cost}", end="\r")
            if init_cost is None:
                init_cost = cost.item()
            cost.backward()
            optimizer.step()
        c_x = x_seq[1].detach()

    with torch.no_grad():
        u_clipped = torch.cat(
            [
                torch.clamp(u_seq[:, :1], -accel_lim, accel_lim),
                torch.clamp(u_seq[:, 1:], -steer_lim, steer_lim),
            ],
            dim=1,
        )
        x_seq = rollout(x0, u_clipped, dt, L)
        cost, final_stage_cost = compute_objective(x_seq, path_points, goal_speed)
        print(f"Final Trajectory Loss: {cost} - Last state loss: {final_stage_cost}")

    return x_seq, u_clipped, cost, init_cost, final_stage_cost


def sample_initial_state_near_path(
    path, max_pos_offset=7.0, max_heading_offset=1.0, max_speed=10.0
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


if __name__ == "__main__":
    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("work", filename_time)
    os.makedirs(out_dir, exist_ok=True)
    for it in range(20):
        T = 5
        dt = 0.1
        # Path to initialise the state
        path = torch.stack(
            [torch.linspace(0, 30, T + 1), torch.zeros(T + 1) + torch.randn(1)], dim=1
        )

        x0 = sample_initial_state_near_path(path)

        # Actual path which is more extended
        path = torch.stack(
            [
                torch.linspace(-60, 90, T + 1),
                torch.zeros(T + 1) + torch.randn(1),
            ],
            dim=1,
        )

        print(f"\nInitial State\n\tX, Y, theta, v:\n{x0}")

        goal_speed = 3.5  # m/s
        L = 1
        steer_lim = 0.5
        accel_lim = 2
        steps = 50
        lr = 0.015
        iters = 200

        x_traj, u_traj, f_cost, i_cost, ff_cost = mpc_optimize(
            x0,
            path,
            goal_speed,
            L=L,
            steer_lim=steer_lim,
            accel_lim=accel_lim,
            T=T,
            lr=lr,
            dt=dt,
            steps=steps,
            iters=iters,
        )

        print(
            f"Last predicted trajectory state x_{len(x_traj)} - ",
            x_traj[-1],
        )
        print(f"Target final state - {[None, path[-1][1].item(), 0, goal_speed]}")

        import matplotlib.pyplot as plt

        # Extract coordinates
        path_x = path[:, 0].numpy()
        path_y = path[:, 1].numpy()

        traj_x = x_traj[:, 0].detach().numpy()
        traj_y = x_traj[:, 1].detach().numpy()

        # distances
        initial_dist = compute_dist(x_traj[0, :2], path).min()
        final_dist = compute_dist(x_traj[-1, :2], path).min()
        # Speeds
        initial_speed = x_traj[0, 3].item()
        final_speed = x_traj[-1, 3].item()
        # alignment errors
        initial_alignment_err = alignment_objective(x_traj[0], path)
        final_alignment_err = alignment_objective(x_traj[-1], path)

        plt.figure(figsize=(10, 6))
        plt.plot(path_x, path_y, "k--", label="Goal Trajectory", linewidth=2)
        plt.plot(
            traj_x[steps:],
            traj_y[steps:],
            "r:",
            label="Future Path",
            linewidth=2,
        )

        # Mark start and end
        for i in range(steps):
            if i < 5:
                plt.scatter(
                    traj_x[i],
                    traj_y[i],
                    s=3.5,
                    color="green",
                    label=f"state {i}",
                    zorder=5,
                )
            else:
                plt.scatter(traj_x[i], traj_y[i], s=3.5, color="green", zorder=5)

        plt.scatter(
            traj_x[-1],
            traj_y[-1],
            s=3.5,
            color="red",
            label=f"last trajectory state x_{T + steps}",
            zorder=5,
        )

        # Annotate
        plt.text(
            traj_x[0],
            traj_y[0] + 0.5,
            f"Start\nSpeed: {initial_speed:.2f} m/s\nY-Dist: {initial_dist:.2f} m"
            f"\Angle Err: {format(initial_alignment_err, '.2f')}"
            f"\nTraj Loss: {format(i_cost, '.2f')}",
            color="green",
            fontsize=7,
            ha="center",
        )

        plt.text(
            traj_x[-1],
            traj_y[-1] + 0.5,
            f"End\nSpeed: {final_speed:.2f} m/s\nY-Dist: {final_dist:.2f} m"
            f"\nAngle Err: {format(final_alignment_err, '.2f')}"
            f"\nTraj Loss: {format(f_cost, '.2f')}\nLast Pred Loss: {format(ff_cost, '.2f')}",
            color="red",
            fontsize=7,
            ha="center",
        )

        info_text = (
            f"vhl length: {L} m\n"
            f"acc limit: {accel_lim}\n"
            f"steer limit: {steer_lim}\n"
            f"Goal Speed: {goal_speed} m/s\n"
            f"Total Steps: {steps}\n"
            f"Horizon Size: {T}\n"
            f"Iterations per step: {iters}\n"
            f"dt: {dt}\n"
            f"optim: adam\n"
            f"lr: {lr}"
        )

        # place in top-left corner of axes coordinates
        plt.text(
            0.02,
            0.98,
            info_text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.title(f"MPC Optimized Trajectory Following - Final Cost: {f_cost}")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{'_'.join([format(x, '.1f') for x in x0])}.png"),
            dpi=100,
        )
