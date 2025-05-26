import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

k1 = 1.3  # /day (3-4) - Infection spread rate
I_max = 10**6  # cells (10**6) - Maximum number of infected cells
k2 = 5 * 10**(-5) # /cells/day (10**(-6)-10**(-4)) - Killing rate of infected cells by CD8 T cells
k3 = 1.0 # /day (1)- Antigen driven CD8 T cell proliferation rate (activation rate)
k4 = 3.0 # /day (2-4) - Antigen driven CD8 T cell suppression rate (exhaustion rate)
kp = 10 # cells (10**(-1)-10**(3)) - Antigen driven proliferation threshold (activation)
ke = 2 * 10**5 # cells (5-2.7*10**4) - Antigen driven suppression threshold (exhaustion)

alpha = 10**(-8) # /cells**2/day - Rate of growth of cytokine pathology
dc = 1 # Rate of loss of cytokine pathology 

def dynamical_motif(y, t):
    I, E = y
    dIdt = k1 * I * (1 - I / I_max) - k2 * I * E
    dEdt = k3 * I * E / (kp + I) - k4 * I * E / (ke + I)
    return [dIdt, dEdt]

I_vals = np.logspace(-5, 0, 50) * I_max  
E_vals = np.logspace(-5, 0, 50) * I_max 
I_mesh, E_mesh = np.meshgrid(I_vals, E_vals)

U = np.zeros(I_mesh.shape)
V = np.zeros(E_mesh.shape)
M = np.zeros(I_mesh.shape)
for i in range(I_mesh.shape[0]):
    for j in range(I_mesh.shape[1]):
        dI, dE = dynamical_motif([I_mesh[i, j], E_mesh[i, j]], 0)
        U[i, j] = dI
        V[i, j] = dE
        M[i, j] = np.hypot(dI, dE)

M[M == 0] = 1
U /= M
V /= M

plt.figure(figsize=(10, 8))
plt.pcolormesh(I_mesh / I_max, E_mesh / I_max, np.log10(M), shading='auto', cmap='plasma', vmin=0, vmax=5)
plt.colorbar(label='log10 speed magnitude')
plt.quiver(I_mesh / I_max, E_mesh / I_max, U, V, color='white', pivot='mid', alpha=0.7)
plt.scatter(np.log10(10**(-6)), np.log10(k1 / k2 / I_max), color='cyan', s=100, edgecolor='black', label='Clearance (I=0)')
plt.scatter(np.log10(1), np.log10(10**(-6)), color='magenta', s=100, edgecolor='black', label='Persistence (I=Imax)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Infected cells (I / Imax)')
plt.ylabel('CD8 T cells (E / Imax)')
plt.show()



def immunopathology(P, t, alpha, dc, I_interp, E_interp):
    I_t = I_interp(t)
    E_t = E_interp(t)
    dPdt = alpha * I_t * E_t - dc * P
    return dPdt

t = np.linspace(0, 20, 1000)
initial_conditions = [
    (0 , 0, 0),
    (10**(-4), 10**(-4), 0),
    (10**(-5), 10**(-3), 0),
    (1, 1, 0),
]

plt.figure(figsize=(10, 6))

for I0, E0, P0 in initial_conditions:
    sol = odeint(dynamical_motif, [I0, E0], t)
    I_vals = sol[:, 0]
    E_vals = sol[:, 1]

    I_interp = interp1d(t, I_vals, fill_value="extrapolate")
    E_interp = interp1d(t, E_vals, fill_value="extrapolate")

    P_sol = odeint(immunopathology, P0, t, args=(alpha, dc, I_interp, E_interp))

    label = f"I0={I0:.0e}, E0={E0:.0e}"
    plt.plot(t, P_sol, label=label)

plt.xlabel('Days post Infection')
plt.ylabel('Cytokine Pathology (A.U)')
plt.legend()
plt.grid(True)
plt.show()

