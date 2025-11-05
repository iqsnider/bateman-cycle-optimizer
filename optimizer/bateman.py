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


if __name__ == '__main__':
    R = 3
    lam1 = np.log(2)/0.894
    lam2 = np.log(2)/4.06

    t_cycle = 3
    sol = simulate_decay(R, lam1, lam2, t_cycle)

    N1, N2 = sol.y
    A1 = lam1*N1
    A2 = lam2*N2

    snr, integral_A1, integral_A2 = snr(sol, lam1, lam2)
    t = sol.t

    # seaborn styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, A1, 'royalblue', linewidth=2.5, label=r'$A_p$ ($^{147}$Ba)')
    ax1.plot(t, A2, 'crimson', linewidth=2.5, label=r'$A_d$ ($^{147}$La)')
    ax1.fill_between(t, A1, alpha=0.3, color='royalblue',
                     label=r'$\int A_p dt$ = {:.2f}'.format(integral_A1))
    ax1.fill_between(t, A2, alpha=0.3, color='crimson',
                     label=r'$\int A_d dt$ = {:.2f}'.format(integral_A2))
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Activity (decays/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Activities with Integrated Areas, SNR = {round(snr, 2)}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()
