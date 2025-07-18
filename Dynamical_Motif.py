import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

k1 = 1.3  # /day (3-4) - Infection spread rate
I_max = 10**6  # cells (10**6) - Maximum number of infected cells
k2 = 5 * 10**(-5) # /cells/day (10**(-6)-10**(-4)) - Killing rate of infected cells by CD8 T cells
k3 = 1.0 # /day (1)- Antigen driven CD8 T cell proliferation rate (activation rate)
k4 = 3.0 # /day (2-4) - Antigen driven CD8 T cell suppression rate (exhaustion rate)
kp = 10 # cells (10**(-1)-10**(3)) - Antigen driven proliferation threshold (activation)
ke = 2 * 10**5 # cells (5-2.7*10**4) - Antigen driven suppression threshold (exhaustion)

alpha = 10**(-8) # /cells**2/day - Rate of growth of cytokine pathology
dc = 1 # Rate of loss of cytokine pathology 
P0 = 0

def steady_state(vars):
    I, E = vars
    dI = k1 * I * (1 - I / I_max) - k2 * I * E
    dE = k3 * I * E / (kp + I) - k4 * I * E / (ke + I)
    return [dI, dE]

def dynamical_motif(y, t):
    I, E, P = y
    dIdt = k1 * I * (1 - I / I_max) - k2 * I * E
    dEdt = k3 * I * E / (kp + I) - k4 * I * E / (ke + I)
    dPdt = alpha * I * E - dc * P
    return [dIdt, dEdt, dPdt]

initial_guess = [2*10**5, 2*10**5]
I_saddle, E_saddle = fsolve(steady_state, initial_guess)
init_saddle = [I_saddle, E_saddle, 0]

tol = 10**(-10)
def distance_from_saddle(E0, I0, direction='forward'):
    y0 = [I0, E0, P0]
    t_span = np.linspace(0, 100, 1000) if direction == 'forward' else np.linspace(0, -100, 1000)
    traj = odeint(dynamical_motif, y0, t_span)
    I_traj, E_traj = traj[:, 0], traj[:, 1]
    d = np.min(np.sqrt((I_traj - I_saddle)**2 + (E_traj - E_saddle)**2))
    return d - tol

Guess_E0_for_I0_0 = 2.029075322800359 * 10**1
Guess_E0_for_I0_Imax = 8.178902 * 10**4
E0_for_I0_0 = fsolve(lambda E0: distance_from_saddle(E0[0], 1, 'forward'), [Guess_E0_for_I0_0])[0]
E0_for_I0_Imax = fsolve(lambda E0: distance_from_saddle(E0[0], I_max, 'backward'), [Guess_E0_for_I0_Imax])[0]

I_vals = np.logspace(-5, 0, 50) * I_max  
E_vals = np.logspace(-5, 0, 50) * I_max 
I_mesh, E_mesh = np.meshgrid(I_vals, E_vals)

U = np.zeros(I_mesh.shape)
V = np.zeros(E_mesh.shape)
M = np.zeros(I_mesh.shape)
for i in range(I_mesh.shape[0]):
    for j in range(I_mesh.shape[1]):
        dI, dE, _ = dynamical_motif([I_mesh[i, j], E_mesh[i, j], P0], 0)
        U[i, j] = dI
        V[i, j] = dE
        M[i, j] = np.hypot(dI, dE)

M[M == 0] = 1
U /= M
V /= M
t = np.linspace(0, 20, 1000)
g = 700

init_above = [1, 7*10**(-4)*I_max, P0]    # Above the basin line (leads to clearance)
init_above_2 = [I_max, E0_for_I0_Imax*2, P0]    # Above the basin line (leads to clearance)
init_above_kiss = [1, E0_for_I0_0*1.05, P0] 
init_above_kiss_2 = [I_max, E0_for_I0_Imax*1.05, P0] 

init_below = [1, I_max*10**(-5)*1.5, P0]    # Below the basin line (leads to persistence)
init_below_2 = [I_max, E0_for_I0_Imax/1.2, P0]    # Below the basin line (leads to persistence)
init_below_kiss = [1, E0_for_I0_0*0.9, P0] 
init_below_kiss_2 = [I_max, E0_for_I0_Imax*0.9, P0] 

init_bound_forward = [1, E0_for_I0_0, P0]     # On the basin boundary (near the saddle)
init_bound_backward = [I_max, E0_for_I0_Imax, P0]     # On the basin boundary (near the saddle)

traj_above = odeint(dynamical_motif, init_above, t)
traj_above_2 = odeint(dynamical_motif, init_above_2, t)
traj_above_kiss = odeint(dynamical_motif, init_above_kiss, t)
traj_above_kiss_2 = odeint(dynamical_motif, init_above_kiss_2, t)

traj_below = odeint(dynamical_motif, init_below, t)
traj_below_2 = odeint(dynamical_motif, init_below_2, t)
traj_below_kiss = odeint(dynamical_motif, init_below_kiss, t)
traj_below_kiss_2 = odeint(dynamical_motif, init_below_kiss_2, t)

traj_saddle = odeint(dynamical_motif, init_saddle, t)
traj_bound_forward = odeint(dynamical_motif, np.float64(init_bound_forward), t)
traj_bound_backward = odeint(dynamical_motif, init_bound_backward, t)

print(E0_for_I0_Imax*2)
print(E0_for_I0_Imax/1.2)


fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# Phase Partrait
pcm = axes[0,0].pcolormesh(I_mesh / I_max, E_mesh / I_max, np.log10(M), shading='auto', cmap='inferno', vmin=0, vmax=5)
fig.colorbar(pcm, ax=axes[0,0], label='log10 magnitude')
axes[0,0].quiver(I_mesh / I_max, E_mesh / I_max, U, V, color='black', pivot='mid', alpha=0.7)
axes[0,0].plot(traj_above[:, 0] / I_max, traj_above[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
axes[0,0].plot(traj_above_2[:, 0] / I_max, traj_above_2[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
axes[0,0].plot(traj_below[:, 0] / I_max, traj_below[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
axes[0,0].plot(traj_below_2[:, 0] / I_max, traj_below_2[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
axes[0,0].scatter(I_saddle / I_max, E_saddle / I_max, color='white')
axes[0,0].plot(traj_bound_forward[:g, 0] / I_max, traj_bound_forward[:g, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
axes[0,0].plot(traj_bound_backward[:g, 0] / I_max, traj_bound_backward[:g, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_xlabel('Infected cells (I / Imax)')
axes[0,0].set_ylabel('CD8 T cells (E / Imax)')

# Cytokine pathology
axes[0,1].plot(t, traj_bound_forward[:, 2], color='black', label='At Saddle point', linewidth=2)
axes[0,1].plot(t, traj_above[:, 2], 'blue', label='Clearance', linewidth=2)
axes[0,1].plot(t, traj_below[:, 2], 'red', label='Persistence', linewidth=2)
axes[0,1].set_xlabel('Time post infection')
axes[0,1].set_ylabel('Cytokine pathology (P)')
axes[0,1].legend(loc='upper left')
axes[0,1].grid(True)
axes[0,1].set_title('Cytokine Pathology Over Time')

# CD8 T cells
axes[1,0].plot(t, traj_bound_forward[:, 1]/I_max, color='black', label='At Saddle point', linewidth=2)
axes[1,0].plot(t, traj_above[:, 1]/I_max, 'blue', label='Clearance', linewidth=2)
axes[1,0].plot(t, traj_below[:, 1]/I_max, 'red', label='Persistence', linewidth=2)
axes[1,0].set_yscale('log')
axes[1,0].set_xlabel('Time post infection')
axes[1,0].set_ylabel('CD8 T cells (E/Imax)')
axes[1,0].set_ylim([10**(-6), 1])
axes[1,0].legend(loc='upper left')
axes[1,0].grid(True)
axes[1,0].set_title('CD8 T Cells Over Time')

# Infected cells
axes[1,1].plot(t, traj_bound_forward[:, 0]/I_max, color='black', label='At Saddle point', linewidth=2)
axes[1,1].plot(t, traj_above[:, 0]/I_max, 'blue', label='Clearance', linewidth=2)
axes[1,1].plot(t, traj_below[:, 0]/I_max, 'red', label='Persistence', linewidth=2)
axes[1,1].set_yscale('log')
axes[1,1].set_xlabel('Time post infection')
axes[1,1].set_ylabel('Infected cells (I/Imax)')
axes[1,1].set_ylim([10**(-6), 1])
axes[1,1].legend(loc='upper left')
axes[1,1].grid(True)
axes[1,1].set_title('Infected Cells Over Time')

plt.tight_layout()
#plt.show()


#########################################################################################

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# Phase Partrait
pcm = axes[0,0].pcolormesh(I_mesh / I_max, E_mesh / I_max, np.log10(M), shading='auto', cmap='inferno', vmin=0, vmax=5)
fig.colorbar(pcm, ax=axes[0,0], label='log10 magnitude')
axes[0,0].quiver(I_mesh / I_max, E_mesh / I_max, U, V, color='black', pivot='mid', alpha=0.7)
axes[0,0].plot(traj_above_kiss[:, 0] / I_max, traj_above_kiss[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
#axes[0,0].plot(traj_above_kiss_2[:, 0] / I_max, traj_above_kiss_2[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
axes[0,0].plot(traj_below_kiss[:, 0] / I_max, traj_below_kiss[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
#axes[0,0].plot(traj_below_kiss_2[:, 0] / I_max, traj_below_kiss_2[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
axes[0,0].scatter(I_saddle / I_max, E_saddle / I_max, color='white')
axes[0,0].plot(traj_bound_forward[:g, 0] / I_max, traj_bound_forward[:g, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
axes[0,0].plot(traj_bound_backward[:g, 0] / I_max, traj_bound_backward[:g, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_xlabel('Infected cells (I / Imax)')
axes[0,0].set_ylabel('CD8 T cells (E / Imax)')

# Cytokine pathology
axes[0,1].plot(t, traj_bound_forward[:, 2], color='black', label='At Saddle point', linewidth=2)
axes[0,1].plot(t, traj_above_kiss[:, 2], 'blue', label='Clearance', linewidth=2)
axes[0,1].plot(t, traj_below_kiss[:, 2], 'red', label='Persistence', linewidth=2)
axes[0,1].set_xlabel('Time post infection')
axes[0,1].set_ylabel('Cytokine pathology (P)')
axes[0,1].legend(loc='upper left')
axes[0,1].grid(True)
axes[0,1].set_title('Cytokine Pathology Over Time')

# CD8 T cells
axes[1,0].plot(t, traj_bound_forward[:, 1]/I_max, color='black', label='At Saddle point', linewidth=2)
axes[1,0].plot(t, traj_above_kiss[:, 1]/I_max, 'blue', label='Clearance', linewidth=2)
axes[1,0].plot(t, traj_below_kiss[:, 1]/I_max, 'red', label='Persistence', linewidth=2)
axes[1,0].set_yscale('log')
axes[1,0].set_xlabel('Time post infection')
axes[1,0].set_ylabel('CD8 T cells (E/Imax)')
axes[1,0].set_ylim([10**(-6), 1])
axes[1,0].legend(loc='upper left')
axes[1,0].grid(True)
axes[1,0].set_title('CD8 T Cells Over Time')

# Infected cells
axes[1,1].plot(t, traj_bound_forward[:, 0]/I_max, color='black', label='At Saddle point', linewidth=2)
axes[1,1].plot(t, traj_above_kiss[:, 0]/I_max, 'blue', label='Clearance', linewidth=2)
axes[1,1].plot(t, traj_below_kiss[:, 0]/I_max, 'red', label='Persistence', linewidth=2)
axes[1,1].set_yscale('log')
axes[1,1].set_xlabel('Time post infection')
axes[1,1].set_ylabel('Infected cells (I/Imax)')
axes[1,1].set_ylim([10**(-6), 1])
axes[1,1].legend(loc='upper left')
axes[1,1].grid(True)
axes[1,1].set_title('Infected Cells Over Time')

plt.tight_layout()
plt.show()