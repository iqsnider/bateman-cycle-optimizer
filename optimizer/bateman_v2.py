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
    # Calculate number of complete cycles
    period = t_duty + t_cycle
    n_cycles = int(np.floor(t_exp / period))

    # Initialize arrays
    t_all = np.array([])
    N1_all = np.array([])
    N2_all = np.array([])

    # Simulate each duty cycle independently (fresh tape each time)
    for cycle in range(n_cycles):
        t_start = cycle*period

        # Time points for this duty cycle
        t_duty_eval = np.linspace(0, t_duty, 1000)

        # Solve from zero initial conditions
        sol_duty = solve_ivp(
            bateman_sys,
            (0, t_duty),
            [0, 0],  # Fresh tape starts at zero
            args=(R, lam1, lam2),
            t_eval=t_duty_eval,
            method="RK45"
        )

        # Store results with absolute time
        t_all = np.concatenate([t_all, t_start + sol_duty.t])
        N1_all = np.concatenate([N1_all, sol_duty.y[0]])
        N2_all = np.concatenate([N2_all, sol_duty.y[1]])

        # Optional: Add flat period for cycle time (populations don't matter)
        if t_cycle > 0 and cycle < n_cycles - 1:  # Don't add after last cycle
            t_all = np.concatenate(
                [t_all, [t_start + t_duty, t_start + period - 1e-9]])
            N1_all = np.concatenate([N1_all, [0, 0]])  # Reset to zero
            N2_all = np.concatenate([N2_all, [0, 0]])  # Reset to zero

    # Create solution object
    sol_experiment = type('obj', (), {})()
    sol_experiment.t = t_all
    sol_experiment.y = np.array([N1_all, N2_all])
    sol_experiment.success = True

    return sol_experiment


# Example usage
R = 10.98*2  # particles per second
lam1 = np.log(2)/0.894
lam2 = np.log(2)/4.06
t_duty = 3  # seconds of beam on
t_cycle = 1  # seconds for tape cycling
t_exp = 57760  # total experiment time in seconds


result = simulate_experiment(R, lam1, lam2, t_duty, t_cycle, t_exp)

A1 = lam1*result.y[0]
A2 = lam2*result.y[1]
integral_A1 = np.trapezoid(A1, result.t)
print(integral_A1)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(result.t, A1, label='A1 (Parent)')
plt.plot(result.t, A2, label='A2 (Daughter)')
plt.xlabel('Time (s)')
plt.ylabel('Activity')
plt.legend()
plt.grid(True)
plt.show()
