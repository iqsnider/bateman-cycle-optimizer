import matplotlib.pyplot as plt
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


def simulate_decay(R, lam1, lam2, t_duty):
    """
    Simulates the Bateman chain decay system
    """
    y0 = [0, 0]  # zero initial conditions

    t_span = (0, t_duty)

    t_eval = np.linspace(0, t_duty, 1000)

    sol = solve_ivp(bateman_sys, t_span, y0, args=(
        R, lam1, lam2), t_eval=t_eval, method="RK45")

    return sol


# def simulate_experiment(R, lam1, lam2, t_duty, t_cycle, t_exp):
#     """
#     Simulates the SuNTAN experiment.
#
#     t_duty: duration before cycling in a new cycle of tape, beam is on (R=R), counts are collected
#     t_cycle: amount of time to cycle in a new section of tape, beam is stopped, counts are not collected
#     t_exp: total experiment time
#     """
#     n_cycles = int(t_exp // (t_duty + t_cycle))
#     t_eval = np.linspace(0, t_exp, 1000*n_cycles)


def simulate_experiment(R, lam1, lam2, t_duty, t_cycle, t_exp):
    """
    Simulates the SuNTAN experiment. 

    t_duty: duration before cycling in a new cycle of tape, beam is on (R=R), counts are collected
    t_cycle: amount of time to cycle in a new section of tape, beam is stopped, counts are not collected
    t_exp: total experiment time
    """
    # number of complete cycles
    period = t_duty + t_cycle
    n_cycles = int(np.floor(t_exp / period))

    t_all = np.array([])
    N1_all = np.array([])
    N2_all = np.array([])

    # each tape is a fresh cycle
    for cycle in range(n_cycles):
        t_start = cycle*period

        t_duty_eval = np.linspace(0, t_duty, 100)

        sol_duty = solve_ivp(
            bateman_sys,
            (0, t_duty),
            [0, 0],  # tape starts at zero
            args=(R, lam1, lam2),
            t_eval=t_duty_eval,
            method="RK45"
        )

        # store results as a function of the actual experiment time
        t_all = np.concatenate([t_all, t_start + sol_duty.t])
        N1_all = np.concatenate([N1_all, sol_duty.y[0]])
        N2_all = np.concatenate([N2_all, sol_duty.y[1]])

        # last cycle handling
        if t_cycle > 0 and cycle < n_cycles - 1:  # don't add after last cycle
            t_all = np.concatenate(
                [t_all, [t_start + t_duty, t_start + period - 1e-9]])
            N1_all = np.concatenate([N1_all, [0, 0]])
            N2_all = np.concatenate([N2_all, [0, 0]])

    # new solution object
    sol_experiment = type('obj', (), {})()
    sol_experiment.t = t_all
    sol_experiment.y = np.array([N1_all, N2_all])
    sol_experiment.success = True

    return sol_experiment


def vary_cycle_time(R, lam1, lam2, t_min, t_max, t_cycle, t_exp, n_cycles_to_test=10):
    """
    Runs the SuNTAN experiment for a varying number of duty cycle times
    """
    duty_times = np.linspace(t_min, t_max, n_cycles_to_test)

    parent_results = []
    daughter_results = []
    for t_duty in duty_times:
        result = simulate_experiment(R, lam1, lam2, t_duty, t_cycle, t_exp)

        A1 = lam1*result.y[0]
        A2 = lam2*result.y[1]
        integral_A1 = np.trapezoid(A1, result.t)
        integral_A2 = np.trapezoid(A2, result.t)

        parent_results.append(integral_A1)
        daughter_results.append(integral_A2)
        print(f"t_duty = {t_duty} Done.")

    return {
        "cycle_times": duty_times,
        "parent_counts": parent_results,
        "daughter_counts": daughter_results,
    }
