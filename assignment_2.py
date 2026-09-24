from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model
from models import num_integrator


FIGURE_DIR = Path("output/assignment_2/figures")

# Control bounds specified by the assignment.
ALPHA_MIN = np.pi / 8
ALPHA_MAX = np.pi / 7
TORQUE_MIN_FACTOR = -0.1
TORQUE_MAX_FACTOR = 0.05
FROUDE_NUMBER = 2.0

# Critically damped (zeta=1) closed loop at omega_n = 4 rad/s: kp = omega_n^2, kd = 2*omega_n.
BALANCE_GAINS = {"kp": 16.0, "kd": 8.0}


def ensure_figure_dir():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR


# ============================================================
# HYBRID SIMULATION (single ballistic step, tau=0)
# ============================================================

def simulate_one_step(initial_state, params, dt=1e-3, max_time=10.0):
    state = np.array(initial_state, dtype=float)
    time = 0.0

    trajectory_time = [time]
    trajectory_state = [state.copy()]

    while time < max_time:
        next_state = num_integrator.rk4(model.evaluate_dynamics, time, state, params, dt)

        if model.event_guard(state, next_state, params):
            post_impact_state = model.event_dynamics(next_state, params)

            trajectory_time.append(time + dt)
            trajectory_state.append(next_state.copy())

            return (
                next_state,
                post_impact_state,
                np.array(trajectory_time),
                np.array(trajectory_state),
            )

        state = next_state
        time += dt

        trajectory_time.append(time)
        trajectory_state.append(state.copy())

    raise RuntimeError("No footstrike occurred within the maximum simulation time.")


def sanity_check_reset(params):
    # expect theta_plus = theta_minus - 2*alpha, theta_dot scaled by cos(2*alpha)
    angle_of_attack = params["angle_of_attack"]
    impact_theta = params["incline"] + angle_of_attack
    test_velocity = 3.0

    test_state = np.array([impact_theta, test_velocity])
    reset_state = model.event_dynamics(test_state, params)

    expected_theta = impact_theta - 2.0 * angle_of_attack
    expected_velocity = test_velocity * np.cos(2.0 * angle_of_attack)

    print("theta after reset:", reset_state[0], "expected:", expected_theta)
    print("theta_dot after reset:", reset_state[1], "expected:", expected_velocity)


def sanity_check_energy(initial_state, params):
    _, _, time, trajectory = simulate_one_step(initial_state, params)

    total_energy = np.array(
        [model.calculate_energy(state, params)[2] for state in trajectory]
    )

    print("energy change during stance:", total_energy[-1] - total_energy[0])

    fig = plt.figure()
    plt.plot(time, total_energy)
    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.title("Energy during continuous stance (ankle_torque = 0)")
    plt.tight_layout()
    fig.savefig(ensure_figure_dir() / "energy.png", dpi=150)
    plt.show()


# ============================================================
# CONTINUOUS BALANCE CONTROLLER (feedback linearization)
# ============================================================

def compute_ankle_torque(state, params, gains):
    """
    Feedback-linearizing, saturated PD torque about the upright equilibrium.

    Cancels the pendulum's gravity term and places the closed-loop response
    directly: theta_ddot = -kp*theta - kd*theta_dot. Saturation (not the
    gains) is what ultimately bounds the region of attraction.
    """
    mass = params["mass"]
    gravity = params["gravity"]
    length = params["length"]

    theta, theta_dot = state
    desired_theta_ddot = -gains["kp"] * theta - gains["kd"] * theta_dot
    torque = mass * length**2 * desired_theta_ddot - mass * gravity * length * np.sin(theta)

    torque_min = TORQUE_MIN_FACTOR * mass * gravity * length
    torque_max = TORQUE_MAX_FACTOR * mass * gravity * length
    return np.clip(torque, torque_min, torque_max)


def fall_angle(params):
    """
    Smallest touchdown angle reachable across the allowed angle-of-attack
    range. Past this angle a footstrike is unavoidable for every valid leg
    placement, so it also serves as the failure boundary while the balance
    controller is holding the stance leg fixed (no footstep).
    """
    return params["incline"] + ALPHA_MIN


def has_fallen(state, params):
    return abs(state[0]) > fall_angle(params)


def simulate_closed_loop(initial_state, params, gains, duration=5.0, dt=2e-3, tol=1e-3):
    """Run the balance controller from initial_state; return whether it converges."""
    state = np.array(initial_state, dtype=float)
    local_params = dict(params)
    time = 0.0

    while time < duration:
        local_params["ankle_torque"] = compute_ankle_torque(state, local_params, gains)
        state = num_integrator.rk4(model.evaluate_dynamics, time, state, local_params, dt)
        time += dt

        if has_fallen(state, local_params):
            return False

    return bool(abs(state[0]) < tol and abs(state[1]) < tol)


# ============================================================
# REGION OF ATTRACTION (grid search)
# ============================================================

def compute_roa_grid(theta_bounds, theta_dot_bounds, resolution, params, gains):
    theta_axis = np.linspace(*theta_bounds, resolution)
    theta_dot_axis = np.linspace(*theta_dot_bounds, resolution)

    roa_grid = np.zeros((resolution, resolution), dtype=bool)
    for i, theta in enumerate(theta_axis):
        for j, theta_dot in enumerate(theta_dot_axis):
            roa_grid[i, j] = simulate_closed_loop([theta, theta_dot], params, gains)

    return theta_axis, theta_dot_axis, roa_grid


def in_roa(state, theta_axis, theta_dot_axis, roa_grid):
    """
    Nearest-neighbor lookup of the balance controller's RoA grid. States
    outside the gridded range are reported as not in the RoA rather than
    clamped to the nearest edge cell, since nearest-neighbor extrapolation
    would otherwise misclassify states far outside the (small) known RoA.
    """
    theta, theta_dot = state
    if not (theta_axis[0] <= theta <= theta_axis[-1]):
        return False
    if not (theta_dot_axis[0] <= theta_dot <= theta_dot_axis[-1]):
        return False

    i = int(np.argmin(np.abs(theta_axis - theta)))
    j = int(np.argmin(np.abs(theta_dot_axis - theta_dot)))
    return bool(roa_grid[i, j])


def plot_roa(theta_axis, theta_dot_axis, roa_grid):
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.pcolormesh(theta_axis, theta_dot_axis, roa_grid.T, cmap="Greens", shading="nearest")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title("Region of attraction of the ankle balance controller")
    fig.savefig(ensure_figure_dir() / "roa.png", dpi=150)
    plt.show()


# ============================================================
# POINCARE SECTION: mid-stance (theta = 0) crossings
# ============================================================

def simulate_stance_phase(theta_dot_k, alpha, params, dt=1e-3, max_time=5.0):
    """
    Simulate one ballistic step (ankle_torque = 0) starting at mid-stance
    (theta = 0) with landing angle-of-attack alpha.

    Returns theta_dot at the next mid-stance crossing, or None if the step
    cannot complete. theta_ddot has the same sign as theta in this model, so
    the forward swing from theta=0 always accelerates towards touchdown; the
    only way a step fails is if the post-impact velocity is too small to
    swing back up through vertical, so the stance leg instead falls further
    backward -- this model defines no backward-impact guard, so that
    trajectory is simply invalid (excluded from the lookup table).
    """
    local_params = dict(params)
    local_params["angle_of_attack"] = alpha
    local_params["ankle_torque"] = 0.0

    state = np.array([0.0, theta_dot_k])
    time = 0.0
    while time < max_time:
        next_state = num_integrator.rk4(model.evaluate_dynamics, time, state, local_params, dt)
        if model.event_guard(state, next_state, local_params):
            state = model.event_dynamics(next_state, local_params)
            break
        state = next_state
        time += dt
    else:
        return None

    time = 0.0
    while time < max_time:
        next_state = num_integrator.rk4(model.evaluate_dynamics, time, state, local_params, dt)
        if state[0] < 0.0 <= next_state[0]:
            return next_state[1]
        if next_state[1] <= 0.0 and next_state[0] < 0.0:
            return None
        state = next_state
        time += dt
    return None


# ============================================================
# STEP-TO-STEP LOOKUP TABLE
# ============================================================

def build_lookup_table(theta_dot_grid, alpha_grid, params):
    table = np.full((theta_dot_grid.size, alpha_grid.size), np.nan)
    for i, theta_dot_k in enumerate(theta_dot_grid):
        for j, alpha in enumerate(alpha_grid):
            result = simulate_stance_phase(theta_dot_k, alpha, params)
            if result is not None:
                table[i, j] = result
    return table


def compute_steps_to_standstill(theta_dot_grid, table, roa_predicate):
    """
    Backward induction ("value iteration") over the lookup table: a grid
    state needs zero more steps if it already lies in the balance
    controller's RoA; otherwise it needs one more step than the best
    reachable state. Iterating this relaxation to a fixed point recovers the
    minimum-steps-to-standstill map and its greedy policy simultaneously.

    Returns (steps, policy): steps[i] is inf if no action sequence from grid
    point i is known to reach standstill; policy[i] is the alpha-grid index
    to use there, or -1 if none is known.
    """
    n_states = theta_dot_grid.size
    n_actions = table.shape[1]

    steps = np.array(
        [0.0 if roa_predicate(theta_dot) else np.inf for theta_dot in theta_dot_grid]
    )
    policy = np.full(n_states, -1, dtype=int)

    changed = True
    while changed:
        changed = False
        for i in range(n_states):
            if steps[i] == 0.0:
                continue

            best_steps = steps[i]
            best_action = policy[i]
            for j in range(n_actions):
                next_theta_dot = table[i, j]
                if np.isnan(next_theta_dot):
                    continue
                nearest = int(np.argmin(np.abs(theta_dot_grid - next_theta_dot)))
                if not np.isfinite(steps[nearest]):
                    continue
                candidate = 1.0 + steps[nearest]
                if candidate < best_steps:
                    best_steps = candidate
                    best_action = j

            if best_steps < steps[i]:
                steps[i] = best_steps
                policy[i] = best_action
                changed = True

    return steps, policy


def longest_convergent_path(start_index, theta_dot_grid, table, steps_to_go, max_depth=50):
    """
    Longest chain of actions from start_index that still reaches standstill
    eventually (only following transitions into states with finite
    steps-to-go), found by depth-first search with a per-path visited set to
    avoid cycles. This is the "maximum number of steps before reaching the
    RoA" -- distinct from, and >=, the greedy/optimal step count.
    """

    def visit(index, visited):
        if steps_to_go[index] == 0.0 or len(visited) > max_depth:
            return 0

        best = 0
        for j in range(table.shape[1]):
            next_theta_dot = table[index, j]
            if np.isnan(next_theta_dot):
                continue
            nearest = int(np.argmin(np.abs(theta_dot_grid - next_theta_dot)))
            if nearest in visited or not np.isfinite(steps_to_go[nearest]):
                continue
            best = max(best, 1 + visit(nearest, visited | {nearest}))
        return best

    return visit(start_index, {start_index})


def check_grid_resolution(alpha_grid, params, roa_predicate, resolutions, reference_theta_dot, theta_dot_max):
    """
    Build the lookup table and steps-to-standstill map at each candidate
    state-grid resolution, returning numbers that show where the answer
    stops changing (used to justify the chosen resolution in the writeup).
    """
    results = []
    for n in resolutions:
        theta_dot_grid = np.linspace(0.05, theta_dot_max, n)
        table = build_lookup_table(theta_dot_grid, alpha_grid, params)
        steps, _ = compute_steps_to_standstill(theta_dot_grid, table, roa_predicate)

        reference_index = int(np.argmin(np.abs(theta_dot_grid - reference_theta_dot)))
        reference_steps = steps[reference_index]
        max_steps = (
            longest_convergent_path(reference_index, theta_dot_grid, table, steps)
            if np.isfinite(reference_steps)
            else None
        )
        results.append(
            {
                "n": n,
                "finite_fraction": float(np.mean(np.isfinite(steps))),
                "reference_theta_dot": float(theta_dot_grid[reference_index]),
                "reference_steps": float(reference_steps),
                "max_steps_from_reference": max_steps,
            }
        )
    return results


def plot_steps_to_standstill(theta_dot_grid, steps):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    finite = np.isfinite(steps)
    ax.scatter(theta_dot_grid[finite], steps[finite], c="#23699b")
    ax.set_xlabel(r"$\dot\theta_k$ at mid-stance (rad/s)")
    ax.set_ylabel("Steps to standstill")
    ax.set_title("Steps to reach the standing RoA vs. initial mid-stance velocity")
    fig.savefig(ensure_figure_dir() / "steps_to_standstill.png", dpi=150)
    plt.show()


# ============================================================
# FULL POLICY TRAJECTORY (walking, then balancing)
# ============================================================

def nearest_index(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def simulate_policy_trajectory(
    initial_theta_dot,
    params,
    gains,
    theta_dot_grid,
    alpha_grid,
    policy,
    theta_axis,
    theta_dot_axis,
    roa_grid,
    dt=1e-4,
    max_time=15.0,
):
    """
    Simulate the full closed-loop walker: choose the landing angle-of-attack
    from the lookup-table policy at each mid-stance crossing (ankle_torque=0)
    until the state enters the balance controller's region of attraction,
    then switch permanently to continuous ankle-torque stabilization.
    """
    local_params = dict(params)
    state = np.array([0.0, initial_theta_dot])

    index = nearest_index(theta_dot_grid, state[1])
    local_params["angle_of_attack"] = alpha_grid[policy[index]] if policy[index] >= 0 else alpha_grid[0]
    local_params["ankle_torque"] = 0.0

    time = 0.0
    time_traj = [time]
    state_traj = [state.copy()]
    completed_steps = 0
    mode = "walking"

    while time < max_time:
        if mode == "walking" and in_roa(state, theta_axis, theta_dot_axis, roa_grid):
            mode = "standing"

        local_params["ankle_torque"] = (
            compute_ankle_torque(state, local_params, gains) if mode == "standing" else 0.0
        )

        next_state = num_integrator.rk4(model.evaluate_dynamics, time, state, local_params, dt)

        if mode == "walking":
            if model.event_guard(state, next_state, local_params):
                next_state = model.event_dynamics(next_state, local_params)
                completed_steps += 1
            elif state[0] < 0.0 <= next_state[0]:
                index = nearest_index(theta_dot_grid, next_state[1])
                if policy[index] >= 0:
                    local_params["angle_of_attack"] = alpha_grid[policy[index]]

        state = next_state
        time += dt
        time_traj.append(time)
        state_traj.append(state.copy())

        if mode == "standing" and abs(state[0]) < 1e-3 and abs(state[1]) < 1e-3:
            break

    return np.array(time_traj), np.array(state_traj), completed_steps


def plot_trajectory(time_traj, state_traj, completed_steps):
    fig, (ax_time, ax_phase) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    ax_time.plot(time_traj, state_traj[:, 0], label=r"$\theta$")
    ax_time.plot(time_traj, state_traj[:, 1], label=r"$\dot\theta$")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_title(f"State history ({completed_steps} footstrikes)")
    ax_time.legend()

    ax_phase.plot(state_traj[:, 0], state_traj[:, 1], color="#23699b")
    ax_phase.plot(state_traj[0, 0], state_traj[0, 1], "o", color="#df8a25", label="Start")
    ax_phase.plot(state_traj[-1, 0], state_traj[-1, 1], "s", color="#2e7d32", label="End")
    ax_phase.set_xlabel(r"$\theta$ (rad)")
    ax_phase.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax_phase.set_title("Phase portrait")
    ax_phase.legend()

    fig.savefig(ensure_figure_dir() / "trajectory.png", dpi=150)
    plt.show()


def animate_walker(state_traj, time_traj, params, completed_steps, fps=25):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(index):
        model.visualize(state_traj[:, index], params, ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s")

    frame_stride = max(1, round(1 / (fps * (time_traj[1] - time_traj[0]))))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)

    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))
    print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
    plt.show()


if __name__ == "__main__":
    params = model.generate_params()
    initial_state = np.array([params["incline"] - params["angle_of_attack"], 1.5])

    sanity_check_reset(params)
    sanity_check_energy(initial_state, params)

    print("\nComputing region of attraction of the ankle balance controller...")
    theta_axis, theta_dot_axis, roa_grid = compute_roa_grid(
        theta_bounds=(-0.09, 0.15),
        theta_dot_bounds=(-0.4, 0.4),
        resolution=41,
        params=params,
        gains=BALANCE_GAINS,
    )
    plot_roa(theta_axis, theta_dot_axis, roa_grid)

    def roa_predicate(theta_dot):
        return in_roa(np.array([0.0, theta_dot]), theta_axis, theta_dot_axis, roa_grid)

    theta_dot_max = np.sqrt(FROUDE_NUMBER * params["gravity"] / params["length"])
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, 15)

    print("\nChecking lookup-table grid resolution...")
    resolution_results = check_grid_resolution(
        alpha_grid,
        params,
        roa_predicate,
        resolutions=[3, 5, 8, 10, 20, 40, 80],
        reference_theta_dot=2.0,
        theta_dot_max=theta_dot_max,
    )
    for result in resolution_results:
        print(result)

    print("\nBuilding step-to-step lookup table and steps-to-standstill map...")
    theta_dot_grid = np.linspace(0.05, theta_dot_max, 40)
    table = build_lookup_table(theta_dot_grid, alpha_grid, params)
    steps, policy = compute_steps_to_standstill(theta_dot_grid, table, roa_predicate)
    plot_steps_to_standstill(theta_dot_grid, steps)

    trapped = int(np.sum(~np.isfinite(steps)))
    print(f"{trapped}/{steps.size} grid states have no known recovering action.")

    qualifying = np.where(np.isfinite(steps) & (steps >= 3.0))[0]
    if qualifying.size == 0:
        qualifying = np.where(np.isfinite(steps))[0]
    chosen_index = int(qualifying[0])
    chosen_theta_dot = float(theta_dot_grid[chosen_index])
    max_steps = longest_convergent_path(chosen_index, theta_dot_grid, table, steps)

    print(f"\nChosen initial condition: theta_dot_0 = {chosen_theta_dot:.4f} rad/s")
    print(f"Optimal steps to standstill: {steps[chosen_index]:.0f}")
    print(f"Maximum steps before entering the RoA: {max_steps}")

    time_traj, state_traj, completed_steps = simulate_policy_trajectory(
        chosen_theta_dot,
        params,
        BALANCE_GAINS,
        theta_dot_grid,
        alpha_grid,
        policy,
        theta_axis,
        theta_dot_axis,
        roa_grid,
    )
    plot_trajectory(time_traj, state_traj, completed_steps)

    display_params = dict(params, angle_of_attack=ALPHA_MIN)
    animate_walker(state_traj.T, time_traj, display_params, completed_steps)
