import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def tape_cycle_plot(t, R, A, A1, A2, N1, N2, integral_A1, integral_A2, snr_vals, integrals_A1, integrals_A2, save):
    # seaborn styling
    A1_minus_A2 = A1 - A2
    max_A1_minus_A2 = max(A1_minus_A2)
    t_max = t[list(A1_minus_A2).index(max_A1_minus_A2)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(t, A1, 'royalblue', linewidth=2.5,
             label=rf'$A_p$ ($^{{{A}}}\mathrm{{Ba}}$)')
    ax1.plot(t, A2, 'crimson', linewidth=2.5,
             label=rf'$A_d$ ($^{{{A}}}\mathrm{{La}}$)')
    ax1.plot(t, A1 - A2, 'orange',
             linewidth=2.5, label=rf'$A_p - A_d$ max = {round(t_max, 2)} s')
    ax1.plot(t_max, max_A1_minus_A2, 'o')
    ax1.fill_between(t, A1, alpha=0.3, color='royalblue',
                     label=r'$\int A_p dt$ = {:.2f}'.format(integral_A1))
    ax1.fill_between(t, A2, alpha=0.3, color='crimson',
                     label=r'$\int A_d dt$ = {:.2f}'.format(integral_A2))
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Activity (decays/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'A = {A} Activities with Integrated Areas, R = {R} pps, SNR = {round(snr_vals[-1], 2)}',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.4)

    ax2.plot(t, snr_vals, 'darkgreen', linewidth=2.5)
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SNR', fontsize=12, fontweight='bold')
    ax2.set_title(r'Signal-to-Noise Ratio $\left(\frac{\int A_p dt}{\int A_d dt}\right)$ Over the Tape Cycle',
                  fontsize=13, fontweight='bold')
    # ax2.set_title(r'Signal-to-Noise Ratio $\left(\frac{A_p}{A_d}\right)$ Over the Tape Cycle',
    #               fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.show()


def plot_vary_cycle_results(results, A, t_exp, parent=None, daughter=None, save=None):
    """
    Plots the mean detected parent & daughter counts vs cycle time,
    styled consistently with tape_cycle_plot, including mass number A,
    and using error bars (no shaded regions).
    """

    plt.style.use('seaborn-v0_8-whitegrid')

    cycle_times = results["cycle_times"]
    parent_counts = results["parent_counts"]
    daughter_counts = results["daughter_counts"]

    # mean & standard deviations
    parent_means = np.array([np.mean(arr) for arr in parent_counts])
    parent_stds = np.array([np.std(arr) for arr in parent_counts])

    daughter_means = np.array([np.mean(arr) for arr in daughter_counts])
    daughter_stds = np.array([np.std(arr) for arr in daughter_counts])

    fig, ax = plt.subplots(figsize=(10, 6))
    parent_label = "Ba"
    daughter_label = "La"
    if parent is not None:
        parent_label = parent
    if daughter is not None:
        daughter_label = daughter

        # parent errorbar curve
    ax.errorbar(
        cycle_times,
        parent_means,
        yerr=parent_stds,
        fmt="-o",
        color="royalblue",
        linewidth=2.5,
        markersize=7,
        capsize=4,
        label=rf"$A_p$ ($^{{{A}}}\mathrm{{{parent_label}}}$) $\beta$-decays"
    )

    # daughter errorbar curve
    ax.errorbar(
        cycle_times,
        daughter_means,
        yerr=daughter_stds,
        fmt="-s",
        color="crimson",
        linewidth=2.5,
        markersize=7,
        capsize=4,
        label=rf"$A_d$ ($^{{{A}}}\mathrm{{{
            daughter_label}}}$) $\beta$-decays"
    )

    ax.set_title(
        rf"Detected $\beta$-decays vs. Cycle Time  (A = {
            A}, Exp Time: {round(t_exp/60/60, 0)} h)",
        fontsize=15,
        fontweight="bold",
        pad=10
    )
    ax.set_xlabel("Cycle Time [s]", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"$\beta$-decays", fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.4)

    ax.legend(
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()
