from optimizer.bateman import simulate_decay, snr, snr_w_time
from optimizer.make_plots import tape_cycle_plot

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer()


@app.command()
def tape(time: float = 3,
         rate: float = 3,
         save: str = None):
    R = rate
    lam1 = np.log(2)/0.894
    lam2 = np.log(2)/4.06

    t_cycle = time
    sol = simulate_decay(R, lam1, lam2, t_cycle)

    N1, N2 = sol.y
    A1 = lam1*N1
    A2 = lam2*N2

    snr_val, integral_A1, integral_A2 = snr(sol, lam1, lam2)
    # snr_vals = snr_w_time(sol, lam1, lam2)
    t = sol.t

    tape_cycle_plot(t, A1, A2, integral_A1, integral_A2, snr_val, save)


if __name__ == '__main__':
    app()
