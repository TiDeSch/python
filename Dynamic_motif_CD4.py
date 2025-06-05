import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

k1 = 1.3  # /day (3-4) - Infection spread rate
I_max = 10**6  # cells (10**6) - Maximum number of infected cells
k2 = 5 * 10**(-5) # /cells/day (10**(-6)-10**(-4)) - Killing rate of infected cells by CD8 T cells
k3 = 0.5 # /day (1)- Antigen driven CD8 T cell proliferation rate (activation rate)
k4 = 3 # /day (2-4) - Antigen driven CD8 T cell suppression rate (exhaustion rate)
kp = 10 # cells (10**(-1)-10**(3)) - Antigen driven proliferation threshold (activation)
ke = 2 * 10**5 # cells (5-2.7*10**4) - Antigen driven suppression threshold (exhaustion)

k7 = k3/2 # /day - CD4 T cell driven proliferation
phiC = kp/5 # Cells - Efficacy threshold of CD4 T cell (half-maximal)
phih = kp*0.9 # Cells - Threshold for CD4 help in boosting CD8 proliferation
gammaC = 0.2 # death rate of CD4 T cells
k3_index = {0.5: 0, 0.8: 1}

alpha = 10**(-8) # /cells**2/day - Rate of growth of cytokine pathology
dc = 1 # Rate of loss of cytokine pathology 

C0 = 0
P0 = 0

def steady_state(vars):
    I, E, C = vars
    dI = k1 * I * (1 - I / I_max) - k2 * I * E
    dE = k3 * (1 + I * E / (kp + I) * (C / (phih + C))) - k4 * I * E / (ke + I)
    dC = k7 * I / (phiC + I) - gammaC * C
    return [dI, dE, dC]

def dynamical_motif(y, t):
    I, E, C, P = y
    dIdt = k1 * I * (1 - I / I_max) - k2 * I * E
    dEdt = k3 * + I * E / (kp + I) * (1 + (C / (phih + C))) - k4 * I * E / (ke + I)
    dCdt = k7 * I / (phiC + I) - gammaC * C
    dPdt = alpha * I * E - dc * P
    return [dIdt, dEdt, dCdt, dPdt]

Guess = [4*10**(-2)*I_max, 7.8*10**(-2)*I_max]
I_saddle_adjust = Guess[k3_index.get(k3, -1)] if k3 in k3_index else 0
initial_guess = [2*10**(-1)*I_max, 2*10**5, C0]
I_saddle, E_saddle, _ = fsolve(steady_state, initial_guess)
I_saddle = I_saddle + I_saddle_adjust
init_saddle = [I_saddle, E_saddle, C0, P0]

tol = 10**(-10)
def distance_from_saddle(E0, I0, direction='forward'):
    y0 = [I0, E0, C0, P0]
    t_span = np.linspace(0, 100, 1000) if direction == 'forward' else np.linspace(0, -100, 1000)
    traj = odeint(dynamical_motif, y0, t_span)
    I_traj, E_traj = traj[:, 0], traj[:, 1]
    d = np.min(np.sqrt((I_traj - I_saddle)**2 + (E_traj - E_saddle)**2))
    return d - tol

Guess_E0_for_I0_0 = 5*10**5
Guess = [1.184851*10**5, 9.38603*10**4]
Guess_E0_for_I0_Imax = Guess[k3_index.get(k3, -1)] if k3 in k3_index else 10**5
E0_for_I0_0 = fsolve(lambda E0: distance_from_saddle(E0[0], 1, 'forward'), [Guess_E0_for_I0_0])[0]
E0_for_I0_Imax = fsolve(lambda E0: distance_from_saddle(E0[0], I_max, 'backward'), [Guess_E0_for_I0_Imax])[0]
print(f"E0 for I0=0: {E0_for_I0_0}")
print(f"E0 for I0=I_max: {E0_for_I0_Imax}")

I_vals = np.logspace(-5, 0, 50) * I_max  
E_vals = np.logspace(-5, 0, 50) * I_max 
I_mesh, E_mesh = np.meshgrid(I_vals, E_vals)

U = np.zeros(I_mesh.shape)
V = np.zeros(E_mesh.shape)
M = np.zeros(I_mesh.shape)
for i in range(I_mesh.shape[0]):
    for j in range(I_mesh.shape[1]):
        dI, dE, _, _ = dynamical_motif([I_mesh[i, j], E_mesh[i, j], C0, P0], 0)
        U[i, j] = dI
        V[i, j] = dE
        M[i, j] = np.hypot(dI, dE)

M[M == 0] = 1
U /= M
V /= M
t = np.linspace(0, 20, 1000)
t_on = np.linspace(0, 15, 1000)

init_above = [1, 10**(-3)*I_max, C0, P0]    # Above the basin line (leads to clearance)
init_above_2 = [I_max, E0_for_I0_Imax*2, C0, P0]    # Above the basin line (leads to clearance)

init_below = [1, I_max*10**(-5)*1.5, C0, P0]    # Below the basin line (leads to persistence)
init_below_2 = [I_max, E0_for_I0_Imax/1.2, C0, P0]    # Below the basin line (leads to persistence)

init_bound_forward = [1, E0_for_I0_0, C0, P0]     # On the basin boundary (near the saddle)
init_bound_backward = [I_max, E0_for_I0_Imax, C0, P0]     # On the basin boundary (near the saddle)

traj_above = odeint(dynamical_motif, init_above, t)
traj_above_2 = odeint(dynamical_motif, init_above_2, t)

traj_below = odeint(dynamical_motif, init_below, t)
traj_below_2 = odeint(dynamical_motif, init_below_2, t)

traj_saddle = odeint(dynamical_motif, init_saddle, t)
traj_bound_forward = odeint(dynamical_motif, init_bound_forward, t_on)
traj_bound_backward = odeint(dynamical_motif, init_bound_backward, t_on)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
pcm = axes[0,0].pcolormesh(I_mesh / I_max, E_mesh / I_max, np.log10(M), shading='auto', cmap='inferno', vmin=0, vmax=5)
fig.colorbar(pcm, ax=axes[0,0], label='log10 magnitude')
axes[0,0].quiver(I_mesh / I_max, E_mesh / I_max, U, V, color='black', pivot='mid', alpha=0.7)
axes[0,0].plot(traj_above[:, 0] / I_max, traj_above[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
axes[0,0].plot(traj_above_2[:, 0] / I_max, traj_above_2[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
axes[0,0].plot(traj_below[:, 0] / I_max, traj_below[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
axes[0,0].plot(traj_below_2[:, 0] / I_max, traj_below_2[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
axes[0,0].scatter(I_saddle / I_max, E_saddle / I_max, color='white')
axes[0,0].plot(traj_bound_forward[:, 0] / I_max, traj_bound_forward[:, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
axes[0,0].plot(traj_bound_backward[:, 0] / I_max, traj_bound_backward[:, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_xlabel('Infected cells (I / Imax)')
axes[0,0].set_ylabel('CD8 T cells (E / Imax)')

# Cytokine pathology
axes[0,1].plot(t, traj_saddle[:, 3], color='black', label='At Saddle point', linewidth=2)
axes[0,1].plot(t, traj_above[:, 3], 'blue', label='Clearance', linewidth=2)
axes[0,1].plot(t, traj_below[:, 3], 'red', label='Persistence', linewidth=2)
axes[0,1].set_xlabel('Time post infection')
axes[0,1].set_ylabel('Cytokine pathology (P)')
axes[0,1].legend()
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
axes[1,0].legend()
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
axes[1,1].legend()
axes[1,1].grid(True)
axes[1,1].set_title('Infected Cells Over Time')

plt.tight_layout()

# CD4 T cells
fig, axes = plt.subplots(1, 1, figsize=(8, 5))
axes.plot(t, traj_bound_forward[:, 2], color='black', label='At Saddle point', linewidth=2)
axes.plot(t, traj_above[:, 2], 'blue', label='Clearance', linewidth=2)
axes.plot(t, traj_below[:, 2], 'red', label='Persistence', linewidth=2)
axes.set_xlabel('Time post infection')
axes.set_ylabel('CD4 T cells (C)')
axes.legend()
axes.grid(True)
axes.set_title('CD4 T Cells Over Time')

plt.show()

