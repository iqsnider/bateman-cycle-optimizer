import numpy as np

from scipy.integrate import solve_ivp


def bateman_sys(t, y, R, lam1, lam2) -> [float, float]:
    """
    Takes a the beam pps rate and the decay constants
    for a parent and daughter nucleus.

    Returns the activties for the two nuclei.
    """
    N1, N2 = y
    dN1dt = R - lam1*N1
    dN2dt = lam1*N1 - lam2*N2

    return [dN1dt, dN2dt]


def simulate_decay(R, lam1, lam2, t_cycle, t_eval=None):
    """
    Simulates the Bateman chain decay system
    """

    y0 = [0, 0]  # zero initial conditions

    t_span = (0, t_cycle)

    # if not evaluating at a specific time
    if t_eval is None:
        t_eval = np.linspace(0, t_cycle, 1000)

    sol = solve_ivp(bateman_sys, t_span, y0, args=(
        R, lam1, lam2), t_eval=t_eval, method="RK45")

    return sol


def snr(sol, lam1, lam2):
    """
    Calculates the Signal-To-Noise ratio between the parent
    and daughter counts.
    """
    N1, N2 = sol.y
    A1 = lam1*N1
    A2 = lam2*N2

    integral_A1 = np.trapezoid(A1, sol.t)
    integral_A2 = np.trapezoid(A2, sol.t)

    snr = integral_A2 / integral_A1 if integral_A1 > 0 else 0

    return snr, integral_A1, integral_A2


def snr_w_time(sol, lam1, lam2):
    """
    Calculates the Signal-To-Noise ratio between the parent
    and daughter counts for each time step.

    returns an array of SNRs
    """
    all_N1, all_N2 = sol.y
    snr_list = []
    integral_A1_list = []
    integral_A2_list = []
    for idx, t in enumerate(sol.t):
        N1 = all_N1[:idx]
        N2 = all_N2[:idx]
        A1 = lam1*N1
        A2 = lam2*N2

        integral_A1 = np.trapezoid(A1, sol.t[:idx])
        integral_A2 = np.trapezoid(A2, sol.t[:idx])

        snr = integral_A1 / integral_A2 if integral_A1 > 0 else 0
        snr_list.append(snr)
        integral_A1_list.append(integral_A1)
        integral_A2_list.append(integral_A2)

    return snr_list, np.array(integral_A1_list), np.array(integral_A2_list)


def monte_carlo(sol, lam1, lam2, eff1=1.0, eff2=1.0, n_samples=100):
    """
    Monte carlo simulation for event activity
    """
    t = sol.t
    N1, N2 = sol.y

    dt = np.diff(t)
    dt = np.append(dt, dt[-1])

    A1 = lam1*N1
    A2 = lam2*N2

    results_parent = np.zeros(n_samples)
    results_daughter = np.zeros(n_samples)

    for i in range(n_samples):
        parent_decayed = np.random.poisson(A1*dt)
        daughter_decayed = np.random.poisson(A2*dt)

        # if not assuming perfect efficiency
        if (eff1 != 1):
            parent_detected = np.random.binomial(parent_decayed, eff1)
        else:
            parent_detected = parent_decayed
        if (eff2 != 1):
            daughter_detected = np.random.binomial(daughter_decayed, eff2)
        else:
            daughter_detected = daughter_decayed

        results_parent[i] = parent_detected.sum()
        results_daughter[i] = daughter_detected.sum()

    return {
        "parent_counts": results_parent,
        "daughter_counts": results_daughter
    }


def simulate_experiment(sol, lam1, lam2, t_cycle, t_exp, eff1=1.0, eff2=1.0, n_samples=100) -> {}:
    n_cycles = int(t_exp // t_cycle)

    results_parent = np.zeros(n_samples)
    results_daughter = np.zeros(n_samples)

    t = sol.t
    N1, N2 = sol.y

    dt = np.diff(t)
    dt = np.append(dt, dt[-1])

    A1 = lam1*N1
    A2 = lam2*N2

    for i in range(n_samples):

        total_parent_detected = []
        total_daughter_detected = []

        for _ in range(n_cycles):
            parent_decayed = np.random.poisson(A1*dt)
            daughter_decayed = np.random.poisson(A2*dt)

            # if not assuming perfect efficiency
            if (eff1 != 1):
                parent_detected = np.random.binomial(parent_decayed, eff1)
            else:
                parent_detected = parent_decayed
            if (eff2 != 1):
                daughter_detected = np.random.binomial(daughter_decayed, eff2)
            else:
                daughter_detected = daughter_decayed

            total_parent_detected.append(parent_detected.sum())
            total_daughter_detected.append(daughter_detected.sum())

        results_parent[i] = np.sum(total_parent_detected)
        results_daughter[i] = np.sum(total_daughter_detected)

    return {
        "parent_counts": results_parent,
        "daughter_counts": results_daughter
    }


def vary_cycle_time(sol, lam1, lam2, t_cycle_range, t_exp, eff1=1.0, eff2=1.0, n_samples=100):
    pass
