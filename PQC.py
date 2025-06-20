import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

p = 1.0       # Monomer production rate
c_T = 0.5     # Transport clearance coefficient
K = 17.0       # Half-saturation constant
h = 2.0       # Hill coefficient

k_n = 0.01    # Aggregation (nucleation + growth) rate
n = 2.0       # Order of monomer in aggregation

def dM_dA_dt(t, y, p, c_T, K, h, k_n, n):
    M, A = y
    clearance = (c_T / (1 + (A / K)**h)) * M
    dMdt = p - clearance - k_n * M**n
    dAdt = k_n * M**n
    return [dMdt, dAdt]

M0 = 0.0  # Initial monomer concentration
A0 = 0.0  # Initial aggregate concentration
y0 = [M0, A0]

t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(
    dM_dA_dt,
    t_span,
    y0,
    args=(p, c_T, K, h, k_n, n),
    t_eval=t_eval
)

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Monomeric Aβ (M)')
plt.plot(sol.t, sol.y[1], label='Insoluble Aggregates (A)')
plt.ylim([0,10])
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Dynamics of Aβ Monomers and Aggregates')
plt.legend()
plt.grid()
plt.show()
