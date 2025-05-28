from scipy.optimize import fsolve
import numpy as np

params = {
    'k1': 1.3, 'k2': 5e-5, 'k3': 1.0, 'k4': 3.0, 'kp': 10,
    'ke': 2e5, 'k7': 0.5, 'I_max': 1e6, 'phiC': 2, 'phih': 9, 'dc': 1
}

k1, k2, k3, k4 = params['k1'], params['k2'], params['k3'], params['k4']
kp, ke = params['kp'], params['ke']
k7, I_max, phiC, phih, dc = params['k7'], params['I_max'], params['phiC'], params['phih'], params['dc']

def steady_state(I):
    I = I[0]
    if I <= 0 or I >= I_max:
        return 1e6
    C = (k7 * I) / (dc * (phiC + I))
    E = (k1 / k2) * (1 - I / I_max)
    lhs = (k3 * I * E / (kp + I)) * (C / (phih + C))
    rhs = (k4 * I * E / (ke + I))
    return lhs - rhs


initial_guesses = [0, I_max, 1e5]
I_steady = [fsolve(steady_state, [guess])[0] for guess in initial_guesses]

E_steady = [(k1 / k2) * (1 - I / I_max) for I in I_steady]
C_steady = [(k7 * I) / (dc * (phiC + I)) for I in I_steady]

results = list(zip(I_steady, E_steady, C_steady))
print(np.array(results))
