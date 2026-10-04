"""Time integrators with the interface used by the assignment examples."""

__all__ = ["explicit_euler", "rk4"]


def explicit_euler(dynamics, time, state, timestep, params):
    """Advance one explicit Euler step without modifying the input state."""
    return state + timestep * dynamics(time, state, params)


def rk4(dynamics, time, state, timestep, params):
    """Advance one classical fourth-order Runge–Kutta step."""
    k1 = dynamics(time, state, params)
    k2 = dynamics(
        time + timestep / 2,
        state + timestep * k1 / 2,
        params,
    )
    k3 = dynamics(
        time + timestep / 2,
        state + timestep * k2 / 2,
        params,
    )
    k4 = dynamics(
        time + timestep,
        state + timestep * k3,
        params,
    )
    return state + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
