import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model


# ============================================================
# NUMERICAL INTEGRATION
# ============================================================

def rk4_step(state, dt, params):
    """
    Take one fourth-order Runge-Kutta integration step.
    """

    k1 = model.dynamics(state, params)

    k2 = model.dynamics(
        state + 0.5 * dt * k1,
        params,
    )

    k3 = model.dynamics(
        state + 0.5 * dt * k2,
        params,
    )

    k4 = model.dynamics(
        state + dt * k3,
        params,
    )

    return state + (
        dt
        / 6.0
        * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    )


def interpolate_impact(
    old_state,
    new_state,
    old_guard,
    new_guard,
):
    """
    Estimate the state at impact by linearly
    interpolating to where the guard equals zero.
    """

    fraction = -old_guard / (
        new_guard - old_guard
    )

    return old_state + fraction * (
        new_state - old_state
    )


# ============================================================
# HYBRID SIMULATION
# ============================================================

def simulate_one_step(
    initial_state,
    params,
    dt=1e-3,
    max_time=10.0,
):
    """
    Simulate the rimless wheel from one post-impact
    state until the next impact.

    Returns:
        pre_impact_state
        post_impact_state
        trajectory_time
        trajectory_state
    """

    state = np.array(
        initial_state,
        dtype=float,
    )

    time = 0.0

    trajectory_time = [time]
    trajectory_state = [state.copy()]

    old_guard = model.detect_impact(
        state,
        params,
    )

    while time < max_time:

        new_state = rk4_step(
            state,
            dt,
            params,
        )

        new_guard = model.detect_impact(
            new_state,
            params,
        )

        # Detect crossing of the impact surface.
        if old_guard < 0.0 and new_guard >= 0.0:

            impact_state = interpolate_impact(
                state,
                new_state,
                old_guard,
                new_guard,
            )

            post_impact_state = model.reset(
                impact_state,
                params,
            )

            trajectory_time.append(
                time + dt
            )

            trajectory_state.append(
                impact_state.copy()
            )

            return (
                impact_state,
                post_impact_state,
                np.array(trajectory_time),
                np.array(trajectory_state),
            )

        state = new_state
        old_guard = new_guard

        time += dt

        trajectory_time.append(time)
        trajectory_state.append(state.copy())

    raise RuntimeError(
        "No impact occurred within the maximum simulation time."
    )


def simulate_steps(
    initial_state,
    params,
    num_steps=50,
    dt=1e-3,
):
    """
    Simulate multiple stance phases and impacts.

    Returns:
        states_after_impact
        states_before_impact
    """

    state = np.array(
        initial_state,
        dtype=float,
    )

    states_before_impact = []
    states_after_impact = []

    for _ in range(num_steps):

        (
            pre_impact,
            post_impact,
            _,
            _,
        ) = simulate_one_step(
            state,
            params,
            dt=dt,
        )

        states_before_impact.append(
            pre_impact.copy()
        )

        states_after_impact.append(
            post_impact.copy()
        )

        state = post_impact

    return (
        np.array(states_after_impact),
        np.array(states_before_impact),
    )


# ============================================================
# SANITY CHECK 1
# IMPACT AND RESET
# ============================================================

def sanity_check_reset(params):
    """
    Check the impact angle and velocity reset.

    Expected:
        theta_minus = gamma + alpha
        theta_plus = gamma - alpha
        theta_dot_plus =
            theta_dot_minus * cos(2 alpha)
    """

    alpha = model.calculate_alpha(params)

    impact_theta = (
        params["slope"] + alpha
    )

    test_velocity = 3.0

    test_state = np.array([
        impact_theta,
        test_velocity,
    ])

    reset_state = model.reset(
        test_state,
        params,
    )

    expected_theta = (
        impact_theta - 2.0 * alpha
    )

    expected_velocity = (
        test_velocity
        * np.cos(2.0 * alpha)
    )

    print("\nRESET SANITY CHECK")
    print("------------------")

    print(
        "theta before:",
        test_state[0],
    )

    print(
        "theta after:",
        reset_state[0],
    )

    print(
        "expected theta after:",
        expected_theta,
    )

    print(
        "theta_dot before:",
        test_state[1],
    )

    print(
        "theta_dot after:",
        reset_state[1],
    )

    print(
        "expected theta_dot after:",
        expected_velocity,
    )


# ============================================================
# SANITY CHECK 2
# ENERGY CONSERVATION DURING STANCE
# ============================================================

def sanity_check_energy(
    initial_state,
    params,
):
    """
    Check that mechanical energy remains approximately
    constant during the continuous stance phase.
    """

    (
        _,
        _,
        time,
        trajectory,
    ) = simulate_one_step(
        initial_state,
        params,
    )

    total_energy = []

    for state in trajectory:

        _, _, energy = (
            model.calculate_energy(
                state,
                params,
            )
        )

        total_energy.append(energy)

    total_energy = np.array(
        total_energy
    )

    energy_change = (
        total_energy[-1]
        - total_energy[0]
    )

    print("\nENERGY SANITY CHECK")
    print("-------------------")

    print(
        "Initial energy:",
        total_energy[0],
    )

    print(
        "Final energy:",
        total_energy[-1],
    )

    print(
        "Energy change during stance:",
        energy_change,
    )

    plt.figure()

    plt.plot(
        time,
        total_energy,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.title(
        "Energy during continuous stance"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# POINCARE RETURN MAP
# ============================================================

def calculate_return_map(
    params,
    velocity_values,
):
    """
    Compute the numerical one-dimensional
    Poincare return map.
    """

    next_velocities = []

    alpha = model.calculate_alpha(
        params
    )

    for velocity in velocity_values:

        # The post-impact angle is gamma - alpha.
        initial_state = np.array([
            params["slope"] - alpha,
            velocity,
        ])

        try:

            (
                _,
                post_impact,
                _,
                _,
            ) = simulate_one_step(
                initial_state,
                params,
            )

            next_velocities.append(
                post_impact[1]
            )

        except RuntimeError:

            next_velocities.append(
                np.nan
            )

    return np.array(
        next_velocities
    )


def plot_return_map(params):
    """
    Plot the numerical return map, identity line,
    and theoretical fixed point.
    """

    fixed_point = (
        model.theoretical_fixed_point(
            params
        )
    )

    velocity_values = np.linspace(
        0.1,
        max(
            4.0,
            1.5 * fixed_point,
        ),
        150,
    )

    next_velocity = (
        calculate_return_map(
            params,
            velocity_values,
        )
    )

    plt.figure()

    plt.plot(
        velocity_values,
        next_velocity,
        label="Numerical return map",
    )

    plt.plot(
        velocity_values,
        velocity_values,
        "--",
        label="Identity line",
    )

    plt.plot(
        fixed_point,
        fixed_point,
        "o",
        label="Fixed point",
    )

    plt.xlabel(
        r"Post-impact $\dot{\theta}_k$"
    )

    plt.ylabel(
        r"Post-impact $\dot{\theta}_{k+1}$"
    )

    plt.title(
        "One-dimensional Poincare return map"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nRETURN MAP")
    print("----------")

    print(
        "Theoretical fixed point:",
        fixed_point,
    )


# ============================================================
# FLOQUET MULTIPLIER
# ============================================================

def estimate_floquet_multiplier(
    params,
    perturbation=1e-4,
):
    """
    Estimate the Floquet multiplier from the
    local slope of the return map.
    """

    fixed_point = (
        model.theoretical_fixed_point(
            params
        )
    )

    alpha = model.calculate_alpha(
        params
    )

    velocity_minus = (
        fixed_point - perturbation
    )

    velocity_plus = (
        fixed_point + perturbation
    )

    state_minus = np.array([
        params["slope"] - alpha,
        velocity_minus,
    ])

    state_plus = np.array([
        params["slope"] - alpha,
        velocity_plus,
    ])

    (
        _,
        next_minus,
        _,
        _,
    ) = simulate_one_step(
        state_minus,
        params,
    )

    (
        _,
        next_plus,
        _,
        _,
    ) = simulate_one_step(
        state_plus,
        params,
    )

    multiplier = (
        next_plus[1]
        - next_minus[1]
    ) / (
        2.0 * perturbation
    )

    return multiplier


# ============================================================
# REGION OF ATTRACTION
# ============================================================

def classify_initial_condition(
    initial_state,
    params,
    num_steps=40,
    tolerance=1e-4,
):
    """
    Classify an initial condition.

    Returns:
        1 -> converges to rolling limit cycle
        0 -> does not converge
    """

    try:

        states_after, _ = simulate_steps(
            initial_state,
            params,
            num_steps=num_steps,
        )

    except RuntimeError:

        return 0

    fixed_point = (
        model.theoretical_fixed_point(
            params
        )
    )

    final_velocities = (
        states_after[-5:, 1]
    )

    error = np.abs(
        final_velocities
        - fixed_point
    )

    if np.all(
        error < tolerance
    ):
        return 1

    return 0


def calculate_roa(
    params,
    theta_values,
    velocity_values,
):
    """
    Calculate the estimated region of attraction
    over a grid of initial conditions.
    """

    roa = np.zeros(
        (
            len(velocity_values),
            len(theta_values),
        )
    )

    for i, velocity in enumerate(
        velocity_values
    ):

        for j, theta in enumerate(
            theta_values
        ):

            initial_state = np.array([
                theta,
                velocity,
            ])

            roa[i, j] = (
                classify_initial_condition(
                    initial_state,
                    params,
                )
            )

    return roa


def plot_roa(params):
    """
    Plot the estimated region of attraction.
    """

    alpha = model.calculate_alpha(
        params
    )

    theta_values = np.linspace(
        params["slope"] - 2.0 * alpha,
        params["slope"] + 2.0 * alpha,
        50,
    )

    velocity_values = np.linspace(
        -3.0,
        3.0,
        60,
    )

    roa = calculate_roa(
        params,
        theta_values,
        velocity_values,
    )

    plt.figure()

    plt.pcolormesh(
        theta_values,
        velocity_values,
        roa,
        shading="auto",
    )

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")

    plt.title(
        "Estimated Region of Attraction"
    )

    plt.colorbar(
        label="Converges to rolling cycle"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# SLOPE SWEEP
# ============================================================

def sweep_slope(params):
    """
    Sweep the slope angle and calculate the Floquet multiplier.

    If the wheel cannot complete a step at a particular slope,
    record NaN for that slope instead of stopping the program.
    """

    slope_degrees = np.linspace(
        1.0,
        12.0,
        12,
    )

    multipliers = []

    for slope_degree in slope_degrees:

        test_params = params.copy()

        test_params["slope"] = np.deg2rad(
            slope_degree
        )

        try:
            multiplier = (
                estimate_floquet_multiplier(
                    test_params
                )
            )

            multipliers.append(
                multiplier
            )

        except RuntimeError:
            print(
                f"Slope {slope_degree:.1f} deg: "
                "no sustained rolling cycle"
            )

            multipliers.append(
                np.nan
            )

    multipliers = np.array(
        multipliers
    )

    plt.figure()

    plt.plot(
        slope_degrees,
        multipliers,
        "o-",
    )

    plt.xlabel(
        "Slope angle (degrees)"
    )

    plt.ylabel(
        "Floquet multiplier"
    )

    plt.title(
        "Floquet Multiplier vs. Slope"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (
        slope_degrees,
        multipliers,
    )


# ============================================================
# NUMBER OF SPOKES SWEEP
# ============================================================

def sweep_spokes(params):
    """
    Sweep the number of spokes from 6 to 12
    and calculate the Floquet multiplier.

    If a rolling cycle cannot be completed,
    record NaN instead of stopping the program.
    """

    spoke_values = np.arange(
        6,
        13,
    )

    multipliers = []

    for number_of_spokes in spoke_values:

        test_params = params.copy()

        test_params["num_spokes"] = int(
            number_of_spokes
        )

        try:
            multiplier = (
                estimate_floquet_multiplier(
                    test_params
                )
            )

            multipliers.append(
                multiplier
            )

        except RuntimeError:
            print(
                f"{number_of_spokes} spokes: "
                "no sustained rolling cycle"
            )

            multipliers.append(
                np.nan
            )

    multipliers = np.array(
        multipliers
    )

    plt.figure()

    plt.plot(
        spoke_values,
        multipliers,
        "o-",
    )

    plt.xlabel(
        "Number of spokes"
    )

    plt.ylabel(
        "Floquet multiplier"
    )

    plt.title(
        "Floquet Multiplier vs. Number of Spokes"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (
        spoke_values,
        multipliers,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    params = model.generate_params()

    alpha = model.calculate_alpha(
        params
    )

    # Start immediately after an impact.
    # Post-impact angle = gamma - alpha.
    initial_state = np.array([
        params["slope"] - alpha,
        1.5,
    ])

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    sanity_check_reset(
        params
    )

    sanity_check_energy(
        initial_state,
        params,
    )

    # --------------------------------------------------------
    # Poincare return map
    # --------------------------------------------------------

    plot_return_map(
        params
    )

    # --------------------------------------------------------
    # Floquet multiplier
    # --------------------------------------------------------

    numerical_multiplier = (
        estimate_floquet_multiplier(
            params
        )
    )

    theoretical_multiplier = (
        model.theoretical_floquet_multiplier(
            params
        )
    )

    print("\nFLOQUET MULTIPLIER")
    print("------------------")

    print(
        "Numerical:",
        numerical_multiplier,
    )

    print(
        "Theoretical:",
        theoretical_multiplier,
    )

    print(
        "Absolute difference:",
        abs(
            numerical_multiplier
            - theoretical_multiplier
        ),
    )

    # --------------------------------------------------------
    # Region of attraction
    # --------------------------------------------------------

    plot_roa(
        params
    )

    # --------------------------------------------------------
    # Slope sweep
    # --------------------------------------------------------

    sweep_slope(
        params
    )

    # --------------------------------------------------------
    # Number of spokes sweep
    # --------------------------------------------------------

    sweep_spokes(
        params
    )