def explicit_euler(dynamics, t, state, params, timestep):
    return state + timestep * dynamics(t, state, params)

def rk4(dynamics, t, state, params, timestep):
    k1 = dynamics(t, state, params)

    k2 = dynamics(
        t + timestep / 2,
        state + timestep * k1 / 2,
        params,
    )

    k3 = dynamics(
        t + timestep / 2,
        state + timestep * k2 / 2,
        params,
    )

    k4 = dynamics(
        t + timestep,
        state + timestep * k3,
        params,
    )

    return state + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)