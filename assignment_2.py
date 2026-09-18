from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model
from models import num_integrator


def simulate_one_step(initial_state, params, dt=1e-3, max_time=10.0):
    state = np.array(initial_state, dtype=float)
    time = 0.0

    trajectory_time = [time]
    trajectory_state = [state.copy()]

    while time < max_time:
        next_state = num_integrator.rk4(model.dynamics, time, state, params, dt)

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

    plt.figure()
    plt.plot(time, total_energy)
    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.title("Energy during continuous stance (ankle_torque = 0)")
    plt.tight_layout()
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

    # still need: stabilizing ankle controller + RoA, Poincare section, lookup table

    dt = 1e-4
    sim_time = 3.0
    desired_number_of_steps = 3

    n_timesteps = round(sim_time / dt) + 1
    time_traj = np.arange(n_timesteps) * dt
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    completed_steps = 0

    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        next_state = num_integrator.rk4(model.dynamics, t, state, params, dt)

        if model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(next_state, params)
            completed_steps += 1

        state_traj[:, step + 1] = next_state
        if completed_steps == desired_number_of_steps:
            break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]

    animate_walker(state_traj, time_traj, params, completed_steps)
