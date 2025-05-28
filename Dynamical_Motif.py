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
t = np.linspace(0, 100, 1000)
t_on = np.linspace(0, 15, 1000)

init_above = [1 * 10**(0), 1 * 10**(2)]    # Above the basin line (leads to clearance)
init_above_2 = [I_max, 1 * 10**(5)]    # Above the basin line (leads to clearance)

init_below = [1 * 10**(0), 7 * 10**0]    # Below the basin line (leads to persistence)
init_below_2 = [I_max, 6 * 10**4]    # Below the basin line (leads to persistence)

init_on = [1 * 10**(0), 2.029075322800359 * 10**1]     # On the basin boundary (near the saddle)
init_on_2 = [I_max, 8.178902 * 10**4]     # On the basin boundary (near the saddle)

traj_above = odeint(dynamical_motif, init_above, t)
traj_above_2 = odeint(dynamical_motif, init_above_2, t)

traj_below = odeint(dynamical_motif, init_below, t)
traj_below_2 = odeint(dynamical_motif, init_below_2, t)

traj_on = odeint(dynamical_motif, init_on, t_on)
traj_on_2 = odeint(dynamical_motif, init_on_2, t_on)

plt.figure(figsize=(10, 8))
plt.pcolormesh(I_mesh / I_max, E_mesh / I_max, np.log10(M), shading='auto', cmap='inferno', vmin=0, vmax=5)
plt.colorbar(label='log10 magnitude')
plt.quiver(I_mesh / I_max, E_mesh / I_max, U, V, color='black', pivot='mid', alpha=0.7)

plt.plot(traj_above[:, 0] / I_max, traj_above[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
plt.plot(traj_above_2[:, 0] / I_max, traj_above_2[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')

plt.plot(traj_below[:, 0] / I_max, traj_below[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
plt.plot(traj_below_2[:, 0] / I_max, traj_below_2[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')

plt.plot(traj_on[:, 0] / I_max, traj_on[:, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
plt.plot(traj_on_2[:, 0] / I_max, traj_on_2[:, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Infected cells (I / Imax)')
plt.ylabel('CD8 T cells (E / Imax)')
plt.show()