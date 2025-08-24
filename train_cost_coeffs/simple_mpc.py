import torch
import torch.nn as nn
import torch.optim as optim


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


# def compute_objective(x_seq, path_points, goal_speed, wp=1.0, ws=1.0):
#     pos_error = ((x_seq[:, :2] - path_points) ** 2).sum(dim=1)  # (T+1,)
#     speed_error = (x_seq[:, 3] - goal_speed) ** 2
#     return (wp * pos_error + ws * speed_error).sum()


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
def compute_objective(x_seq, path_points, goal_speed, wp=0.15, ws=0.5, wa=1.0):
    pos_error = []
    angle_error = []
    for i in range(x_seq.shape[0]):
        dists = compute_dist(x_seq[i], path_points)
        min_id = torch.argmin(dists)
        pos_error.append(dists[torch.clip(min_id - 4, 0) : min_id + 4].sum() / 9)
        angle_error.append(alignment_objective(x_seq[i], path))
    pos_error = torch.stack(pos_error)
    angle_error = torch.stack(angle_error)
    speed_error = (x_seq[:, 3] - goal_speed) ** 2
    return (wp * pos_error + ws * speed_error + wa * angle_error).sum()


# Fix objective function:
def compute_dist(x, path_points):
    # dists = ((path_points - x) ** 2).sum(dim=1)
    dists = (path_points[:, 1] - x[1]) ** 2
    return dists


def mpc_optimize(
    x0,
    path_points,
    goal_speed,
    T=20,
    dt=0.1,
    L=1,
    steer_lim=0.5,
    accel_lim=2.0,
    lr=0.007,
    iters=200,
):
    """
    x0: (1, 4)
    path_points: (T+1, 2)
    """

    u_seq = torch.zeros(T, 2, requires_grad=True)  # [a, δ]
    optimizer = optim.Adam([u_seq], lr=lr)

    for i in range(iters):
        optimizer.zero_grad()

        u_clipped = torch.cat(
            [
                torch.clamp(u_seq[:, :1], -accel_lim, accel_lim),
                torch.clamp(u_seq[:, 1:], -steer_lim, steer_lim),
            ],
            dim=1,
        )

        x_seq = rollout(x0, u_clipped, dt, L)
        cost = compute_objective(x_seq, path_points, goal_speed)
        print(f"Cost: {cost}", end="\r")
        cost.backward()
        optimizer.step()

    # Final trajectory and clipped controls
    with torch.no_grad():
        u_clipped = torch.cat(
            [
                torch.clamp(u_seq[:, :1], -accel_lim, accel_lim),
                torch.clamp(u_seq[:, 1:], -steer_lim, steer_lim),
            ],
            dim=1,
        )
        x_seq = rollout(x0, u_clipped, dt, L)

    return x_seq, u_clipped, cost


def generate_s_curve_path(length=30.0, T=30, amplitude=3.0, frequency=0.2):
    """
    Generates a smooth S-curve path: x is linear, y is sinusoidal.

    Args:
        length: total length along x-axis
        T: number of steps (for MPC horizon)
        amplitude: peak height of sine curve (in meters)
        frequency: controls how many waves in the given length

    Returns:
        path: (T+1, 2) tensor with x, y coordinates
    """
    x_vals = torch.linspace(0, length, T + 1)
    y_vals = amplitude * torch.sin(frequency * x_vals)
    return torch.stack([x_vals, y_vals], dim=1)  # (T+1, 2)


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


if __name__ == "__main__":
    for it in range(20):
        T = 50
        dt = 0.1
        # Straight path 30 meters ahead
        path = torch.stack(
            [torch.linspace(0, 30, T + 1), torch.zeros(T + 1) + torch.randn(1)], dim=1
        )

        path = torch.stack(
            [
                torch.linspace(0, 30, T + 1),
                torch.zeros(T + 1) + torch.randn(1),
                # torch.zeros(T + 1) - 0.4768,
            ],
            dim=1,
        )
        # path = generate_s_curve_path()
        # x0 = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # X, Y, θ, v
        x0 = sample_initial_state_near_path(path)
        # x0 = torch.Tensor([28.5298, 2.5356, 0.9224, 1.3335])
        print(f"\nInitial State: X, Y, theta, v: {x0}")
        goal_speed = 2.5  # m/s

        x_traj, u_traj, f_cost = mpc_optimize(x0, path, goal_speed, T=T, dt=dt)

        print("Final state: X, Y, theta, v", x_traj[-1])
        print(
            f"TARGET state: X, Y, theta, v: {torch.tensor((path[-1][0], path[-1][1], 0, goal_speed))} (except X)"
        )

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
        plt.plot(path_x, path_y, "k--", label="Reference Path", linewidth=2)
        plt.plot(traj_x, traj_y, "r-", label="Optimized Trajectory", linewidth=2)

        # Mark start and end
        plt.scatter(traj_x[0], traj_y[0], color="green", label="Start", zorder=5)
        plt.scatter(traj_x[-1], traj_y[-1], color="blue", label="End", zorder=5)

        # Annotate
        plt.text(
            traj_x[0],
            traj_y[0] + 0.5,
            f"Start\nSpeed: {initial_speed:.2f} m/s\nDist: {initial_dist:.2f} m"
            f"\nAlignment Err: {format(initial_alignment_err, '.2f')}",
            color="green",
            fontsize=9,
            ha="center",
        )

        plt.text(
            traj_x[-1],
            traj_y[-1] + 0.5,
            f"End\nSpeed: {final_speed:.2f} m/s\nDist: {final_dist:.2f} m"
            f"\nAlignment Error: {format(final_alignment_err, '.2f')}",
            color="blue",
            fontsize=9,
            ha="center",
        )

        plt.title(f"MPC Optimized Trajectory Following - Final Cost: {f_cost}")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{'_'.join([format(x, '.1f') for x in x0])}.png", dpi=100)
