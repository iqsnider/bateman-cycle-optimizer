from optimizer.bateman import simulate_decay, snr

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer()


@app.command()
def tape(save: str = None):
    R = 3
    lam1 = np.log(2)/0.894
    lam2 = np.log(2)/4.06

    t_cycle = 3
    sol = simulate_decay(R, lam1, lam2, t_cycle)

    N1, N2 = sol.y
    A1 = lam1*N1
    A2 = lam2*N2

    snr_val, integral_A1, integral_A2 = snr(sol, lam1, lam2)
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
    ax1.set_title(f'Activities with Integrated Areas, SNR = {round(snr_val, 2)}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    app()
