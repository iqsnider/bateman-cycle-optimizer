from optimizer.bateman import simulate_decay, snr, snr_w_time, monte_carlo, simulate_experiment, vary_cycle_time
from optimizer.make_plots import tape_cycle_plot

import numpy as np
import typer

app = typer.Typer()


@app.command()
def tape(time: float = 3,
         rate: float = 3,
         tp: float = 0.894,
         td: float = 4.06,
         A: int = 147,
         save: str = None):

    R = rate
    lam1 = np.log(2)/tp
    lam2 = np.log(2)/td

    t_cycle = time
    sol = simulate_decay(R, lam1, lam2, t_cycle)

    N1, N2 = sol.y
    A1 = lam1*N1
    A2 = lam2*N2

    snr_val, integral_A1, integral_A2 = snr(sol, lam1, lam2)
    snr_vals, integrals_A1, integrals_A2 = snr_w_time(sol, lam1, lam2)
    t = sol.t

    tape_cycle_plot(t, R, A, A1, A2, N1, N2, integral_A1,
                    integral_A2, snr_vals, integrals_A1, integrals_A2, save)


@app.command()
def mc(time: float = 3,
       rate: float = 3,
       tp: float = 0.894,
       td: float = 4.06,
       eff1: float = 1.0,
       eff2: float = 1.0,
       samples: int = 500):

    lam1 = np.log(2)/tp
    lam2 = np.log(2)/td
    sol = simulate_decay(rate, lam1, lam2, time)

    mc_res = monte_carlo(sol, lam1, lam2, eff1, eff2, samples)
    print(f"\nMonte Carlo Results (samples: {samples})\n")
    print(f"Mean parent counts:   {mc_res['parent_counts'].mean():.2f}")
    print(f"Mean daughter counts: {mc_res['daughter_counts'].mean():.2f}")
    print(f"Std parent counts:    {mc_res['parent_counts'].std():.2f}")
    print(f"Std daughter counts:  {mc_res['daughter_counts'].std():.2f}")


@app.command()
def mc_exp(t_cycle: float = 3,
           t_exp: float = 28800,
           rate: float = 3,
           tp: float = 0.894,
           td: float = 4.06,
           eff1: float = 1.0,
           eff2: float = 1.0,
           samples: int = 10,
           A: int = 147,
           save: str = None):

    lam1 = np.log(2)/tp
    lam2 = np.log(2)/td
    sol = simulate_decay(rate, lam1, lam2, t_cycle)

    mc_res = simulate_experiment(
        sol, lam1, lam2, t_cycle, t_exp, eff1, eff2, samples)
    print(f"\nA = {A} Monte Carlo Results (Cycle time: {
          t_cycle} s, experiment time: {round(t_exp/60/60, 0)} hrs, samples: {samples})\n")
    print(f"Mean parent counts:   {mc_res['parent_counts'].mean():.2f}")
    print(f"Mean daughter counts: {mc_res['daughter_counts'].mean():.2f}")
    print(f"Std parent counts:    {mc_res['parent_counts'].std():.2f}")
    print(f"Std daughter counts:  {mc_res['daughter_counts'].std():.2f}")


@app.command()
def mc_search(t_min: float = 0.5,
              t_max: float = 20,
              t_exp: float = 28800,
              bins: int = 10,
              rate: float = 3,
              tp: float = 0.894,
              td: float = 4.06,
              eff1: float = 1.0,
              eff2: float = 1.0,
              samples: int = 10,
              A: int = 147,
              save: str = None):

    lam1 = np.log(2)/tp
    lam2 = np.log(2)/td

    mc_res = vary_cycle_time(rate, lam1, lam2, t_min, t_max,
                             t_exp, eff1, eff2, samples, n_cycles_to_test=bins)


if __name__ == '__main__':
    app()
