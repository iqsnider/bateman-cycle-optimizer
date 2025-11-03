import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    R = 3
    lam1 = np.log(2)/0.894
    lam2 = np.log(2)/4.06

    t_cycle = 3
    sol = simulate_decay(R, lam1, lam2, t_cycle)

    N1, N2 = sol.y
    A1 = lam1*N1
    A2 = lam2*N2

    # seaborn styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # activities
    ax1.plot(sol.t, A1, 'royalblue', linewidth=2.5,
             label=r'$A_p$ ($^{147}$Ba)')
    ax1.plot(sol.t, A2, 'crimson', linewidth=2.5, label=r'$A_d$ ($^{147}$La)')
    ax1.set_ylabel('Activity (decays/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Parent and Daughter Activities vs Time',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # populations
    ax2.plot(sol.t, N1, 'royalblue', linestyle='--',
             linewidth=2.5, label=r'$N_p$ ($^{147}$Ba)')
    ax2.plot(sol.t, N2, 'crimson', linestyle='--',
             linewidth=2.5, label=r'$N_d$ ($^{147}$La)')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Population', fontsize=12, fontweight='bold')
    ax2.set_title('Nuclear Populations vs Time',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()
