import numpy as np
from scipy.optimize import root
import sympy as sp

k1 = 1.3          # /day
I_max = 10**6       # cells
k2 = 5*10**(-5)         # /cells/day
k3 = 1.0          # /day
k4 = 3.0          # /day
kp = 10           # cells
ke = 2*10**5          # cells
f = 0.9

def steady_state(X):
    I, E = X
    dIdt = k1 * I * (1 - I / I_max) - k2 * I * E
    dEdt = (k3 * I * E / (kp + I)) * (1 - f) - (k4 * I * E / (ke + I))
    return [dIdt, dEdt]


initial_guess = [10**(-2)*I_max, 10**(-2)*I_max]
sol = root(steady_state, initial_guess, method='hybr')

I_ss, E_ss = sol.x
if sol.success and 0 <= I_ss <= I_max and 0 <= E_ss <= 1e6:
    print(f"Numerical steady-state found:\nI = {I_ss:.2f} cells\nE = {E_ss:.2f} cells")
else:
    print("No valid steady-state found in the specified range.")



#####################################################
I, E = sp.symbols('I E', real=True, positive=True)
k1, k2, k3, k4, kp, ke, f, I_max = sp.symbols('k1 k2 k3 k4 kp ke f I_max', real=True, positive=True)

dIdt = k1 * I * (1 - I / I_max) - k2 * I * E
dEdt = (k3 * I * E / (kp + I)) * (1 - f) - (k4 * I * E / (ke + I))

steady_state_eqs = [sp.Eq(dIdt, 0), sp.Eq(dEdt, 0)]

steady_states = sp.solve(steady_state_eqs, (I, E), dict=True)
for i, sol in enumerate(steady_states):
    print(f"Steady State {i+1}:")
    print(sol)