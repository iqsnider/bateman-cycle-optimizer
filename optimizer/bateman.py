import numpy as np

from scipy.integrate import solve_ivp


def bateman_sys(R, lam1, lam2, N1, N2) -> np.ndarry:
    """
    Takes a the beam pps rate and the decay constants
    for a parent and daughter nucleus.

    Returns the activties for the two nuclei.
    """
    dN1dt = R - lam1*N1
    dN2dt = lam1*N1 - lam2*N2

    return [dN1dt, dN2dt]
