import numpy as np


def generate_params():
    """Return default rimless-wheel parameters."""
    return {
        "gravity": 9.81,
        "length": 1.0,
        "mass": 1.0,
        "slope": np.deg2rad(5.0),
        "num_spokes": 8,
    }


def calculate_alpha(params):
    """Return alpha, where 2*alpha is the angle between spokes."""
    return np.pi / params["num_spokes"]


def dynamics(state, params):
    """
    Continuous rimless-wheel dynamics.

    state[0] = theta
    state[1] = theta_dot
    """
    gravity = params["gravity"]
    length = params["length"]

    theta = state[0]
    theta_dot = state[1]

    theta_ddot = gravity / length * np.sin(theta)

    return np.array([
        theta_dot,
        theta_ddot,
    ])


def detect_impact(state, params):
    """
    Return the impact guard value.

    Impact occurs when:
        theta = gamma + alpha
    """
    slope = params["slope"]
    alpha = calculate_alpha(params)

    return state[0] - (slope + alpha)


def reset(state, params):
    """
    Apply the impact/reset map.

    theta_plus = theta_minus - 2*alpha
    theta_dot_plus = theta_dot_minus*cos(2*alpha)
    """
    alpha = calculate_alpha(params)

    theta_minus = state[0]
    theta_dot_minus = state[1]

    theta_plus = theta_minus - 2.0 * alpha
    theta_dot_plus = theta_dot_minus * np.cos(2.0 * alpha)

    return np.array([
        theta_plus,
        theta_dot_plus,
    ])


def calculate_energy(state, params):
    """Return kinetic, potential, and total mechanical energy."""
    mass = params["mass"]
    gravity = params["gravity"]
    length = params["length"]

    theta = state[0]
    theta_dot = state[1]

    kinetic = 0.5 * mass * length**2 * theta_dot**2
    potential = mass * gravity * length * np.cos(theta)

    return kinetic, potential, kinetic + potential


def theoretical_return_map(angular_velocity, params):
    """
    Analytical one-step return map.

    This is used to validate the numerical simulation.
    """
    gravity = params["gravity"]
    length = params["length"]
    slope = params["slope"]
    alpha = calculate_alpha(params)

    impact_factor = np.cos(2.0 * alpha)

    energy_gain = (
        4.0
        * gravity
        / length
        * np.sin(slope)
        * np.sin(alpha)
    )

    return impact_factor * np.sqrt(
        angular_velocity**2 + energy_gain
    )


def theoretical_fixed_point(params):
    """Return the analytical post-impact fixed point."""
    gravity = params["gravity"]
    length = params["length"]
    slope = params["slope"]
    alpha = calculate_alpha(params)

    c = np.cos(2.0 * alpha)

    numerator = (
        c**2
        * 4.0
        * gravity
        / length
        * np.sin(slope)
        * np.sin(alpha)
    )

    denominator = 1.0 - c**2

    return np.sqrt(numerator / denominator)


def theoretical_floquet_multiplier(params):
    """Return the analytical Floquet multiplier."""
    alpha = calculate_alpha(params)

    return np.cos(2.0 * alpha) ** 2