import numpy as np


def L2_diff(apotential, apppotential, gcenter):
    pot_diff = apotential - apppotential
    N = gcenter.size
    pot_diff_norm = np.sqrt(np.sum(np.square(pot_diff)/N))
    return pot_diff_norm
