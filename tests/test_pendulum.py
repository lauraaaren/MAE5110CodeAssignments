import numpy as np

from integrators import rk4
from models import pendulum

TIMESTEP = 0.01
STEPS = 100


def _energy_over_steps(initial_state, params):
    state = np.array(initial_state, dtype=float)
    energies = [sum(pendulum.calculate_energy(state, params))]

    for step in range(STEPS):
        state = rk4(pendulum.dynamics, step * TIMESTEP, state, TIMESTEP, params)
        energies.append(sum(pendulum.calculate_energy(state, params)))

    return np.array(energies)


def test_unforced_undamped_pendulum_conserves_energy():
    params = pendulum.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0

    energies = _energy_over_steps([0.2, 0.3], params)

    assert np.all(np.isclose(energies, energies[0], rtol=1e-6, atol=1e-8))


def test_torque_changes_angular_acceleration():
    params = pendulum.generate_params()
    params["mass"] = 2.0
    params["length"] = 0.5
    params["damping_coeff"] = 0.0
    state = np.array([0.0, 0.0])

    for torque in (-1.0, 1.0):
        params["torque"] = torque
        angular_acceleration = pendulum.dynamics(0.0, state, params)[1]
        expected = torque / (params["mass"] * params["length"] ** 2)
        assert np.isclose(angular_acceleration, expected)


def test_damping_dissipates_energy():
    params = pendulum.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.4

    energies = _energy_over_steps([0.2, 1.0], params)

    assert np.all(np.diff(energies) <= 1e-8)
    assert energies[-1] < energies[0] - 0.01
