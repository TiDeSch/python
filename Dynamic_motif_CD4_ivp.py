import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

k1 = 1.3  # /day (3-4) - Infection spread rate
I_max = 10**6  # cells (10**6) - Maximum number of infected cells
k2 = 5 * 10**(-5) # /cells/day (10**(-6)-10**(-4)) - Killing rate of infected cells by CD8 T cells
k3 = 1 # /day (1)- Antigen driven CD8 T cell proliferation rate (activation rate)
k4 = 3.0 # /day (2-4) - Antigen driven CD8 T cell suppression rate (exhaustion rate)
kp = 10 # cells (10**(-1)-10**(3)) - Antigen driven proliferation threshold (activation)
ke = 2 * 10**5 # cells (5-2.7*10**4) - Antigen driven suppression threshold (exhaustion)

k7 = k3/2 # /day - CD4 T cell driven proliferation
phiC = 2 # Cells - Efficacy threshold of CD4 T cell (half-maximal)
phih = 9 # Cells - Threshold for CD4 help in boosting CD8 proliferation
gammaC = 0.2 # death rate of CD4 T cells

alpha = 10**(-8) # /cells**2/day - Rate of growth of cytokine pathology
dc = 1 # Rate of loss of cytokine pathology 

C0 = 0
P0 = 0

def steady_state(vars):
    I, E, C = vars
    dI = k1 * I * (1 - I / I_max) - k2 * I * E
    dE = k3 * I * E / (kp + I) * (C / (phih + C)) - k4 * I * E / (ke + I)
    dC = k7 * I / (phiC + I) - gammaC * C
    return [dI, dE, dC]

def dynamical_motif(t, y):  # Note: t and y are swapped for solve_ivp
    I, E, C, P = y
    dIdt = k1 * I * (1 - I / I_max) - k2 * I * E
    dEdt = k3 * I * E / (kp + I) * (C / (phih + C)) - k4 * I * E / (ke + I)
    dCdt = k7 * I / (phiC + I) - gammaC * C
    dPdt = alpha * I * E - dc * P
    return [dIdt, dEdt, dCdt, dPdt]

initial_guess = [3.559*10**3, 2.591*10**4, 0]
I_saddle, E_saddle, _ = fsolve(steady_state, initial_guess)
delta = 0
init_saddle = [I_saddle * (1 + delta), E_saddle * (1 + delta), C0, P0]

tol = 10**(-20)
def distance_from_saddle(E0, I0, direction='forward'):
    y0 = [I0, E0, C0, P0]
    if direction == 'forward':
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 1000)
    else:  # backward
        t_span = (0, -100)
        t_eval = np.linspace(0, -100, 1000)
    
    try:
        sol = solve_ivp(dynamical_motif, t_span, y0, t_eval=t_eval, dense_output=True, 
                       method='RK45', rtol=1e-8, atol=1e-10)
        if sol.success:
            I_traj, E_traj = sol.y[0], sol.y[1]
            d = np.min(np.sqrt((I_traj - I_saddle)**2 + (E_traj - E_saddle)**2))
            return d - tol
        else:
            return float('inf')  # Return large value if integration failed
    except:
        return float('inf')  # Return large value if integration failed

Guess_E0_for_I0_0 = 1.35479 * 10**4
Guess_E0_for_I0_Imax = 1.6514083 * 10**5
E0_for_I0_0 = fsolve(lambda E0: distance_from_saddle(E0[0], 1, 'forward'), [Guess_E0_for_I0_0])[0]
E0_for_I0_Imax = fsolve(lambda E0: distance_from_saddle(E0[0], I_max, 'backward'), [Guess_E0_for_I0_Imax])[0]
print(f"E0 for I0=0: {E0_for_I0_0:.6e}")
print(f"E0 for I0=I_max: {E0_for_I0_Imax:.6e}")

I_vals = np.logspace(-5, 0, 50) * I_max  
E_vals = np.logspace(-5, 0, 50) * I_max 
I_mesh, E_mesh = np.meshgrid(I_vals, E_vals)

U = np.zeros(I_mesh.shape)
V = np.zeros(E_mesh.shape)
M = np.zeros(I_mesh.shape)

for i in range(I_mesh.shape[0]):
    for j in range(I_mesh.shape[1]):
        dI, dE, _, _ = dynamical_motif(0, [I_mesh[i, j], E_mesh[i, j], C0, P0])  # Note: t=0 as first argument
        U[i, j] = dI
        V[i, j] = dE
        M[i, j] = np.hypot(dI, dE)

M[M == 0] = 1
U /= M
V /= M
t = np.linspace(0, 20, 1000)
t_on = np.linspace(0, 20, 1000)

init_above = [1, E0_for_I0_0*2.7, C0, P0]    # Above the basin line (leads to clearance)
init_above_2 = [I_max, E0_for_I0_Imax*1.2, C0, P0]    # Above the basin line (leads to clearance)

init_below = [1, E0_for_I0_0/10.7, C0, P0]    # Below the basin line (leads to persistence)
init_below_2 = [I_max, E0_for_I0_Imax/1.2, C0, P0]    # Below the basin line (leads to persistence)

init_bound_forward = [1, E0_for_I0_0, C0, P0]     # On the basin boundary (near the saddle)
init_bound_backward = [I_max, E0_for_I0_Imax, C0, P0]     # On the basin boundary (near the saddle)

# Convert odeint calls to solve_ivp
sol_above = solve_ivp(dynamical_motif, (0, 20), init_above, t_eval=t, dense_output=True)
traj_above = sol_above.y.T

sol_above_2 = solve_ivp(dynamical_motif, (0, 20), init_above_2, t_eval=t, dense_output=True)
traj_above_2 = sol_above_2.y.T

sol_below = solve_ivp(dynamical_motif, (0, 20), init_below, t_eval=t, dense_output=True)
traj_below = sol_below.y.T

sol_below_2 = solve_ivp(dynamical_motif, (0, 20), init_below_2, t_eval=t, dense_output=True)
traj_below_2 = sol_below_2.y.T

sol_saddle = solve_ivp(dynamical_motif, (0, 20), init_saddle, t_eval=t, dense_output=True)
traj_saddle = sol_saddle.y.T

sol_bound_forward = solve_ivp(dynamical_motif, (0, 20), init_bound_forward, t_eval=t_on, dense_output=True)
traj_bound_forward = sol_bound_forward.y.T

sol_bound_backward = solve_ivp(dynamical_motif, (0, 20), init_bound_backward, t_eval=t_on, dense_output=True)
traj_bound_backward = sol_bound_backward.y.T

plt.figure(figsize=(10, 8))
plt.pcolormesh(I_mesh / I_max, E_mesh / I_max, np.log10(M), shading='auto', cmap='inferno', vmin=0, vmax=5)
plt.colorbar(label='log10 magnitude')
plt.quiver(I_mesh / I_max, E_mesh / I_max, U, V, color='black', pivot='mid', alpha=0.7)

plt.plot(traj_above[:, 0] / I_max, traj_above[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')
plt.plot(traj_above_2[:, 0] / I_max, traj_above_2[:, 1] / I_max, 'blue', lw=2, label='Above basin (Clearance)')

plt.plot(traj_below[:, 0] / I_max, traj_below[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')
plt.plot(traj_below_2[:, 0] / I_max, traj_below_2[:, 1] / I_max, 'red', lw=2, label='Below basin (Persistence)')

plt.scatter(I_saddle / I_max, E_saddle / I_max, color='white')
plt.plot(traj_bound_forward[:, 0] / I_max, traj_bound_forward[:, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')
plt.plot(traj_bound_backward[:, 0] / I_max, traj_bound_backward[:, 1] / I_max, 'white', lw=2, linestyle='--', label='On basin (Saddle)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Infected cells (I / Imax)')
plt.ylabel('CD8 T cells (E / Imax)')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Cytokine pathology
axes[0, 0].plot(t, traj_saddle[:, 3], color='black', label='At Saddle point', linewidth=2)
axes[0, 0].plot(t, traj_above[:, 3], 'blue', label='Clearance', linewidth=2)
axes[0, 0].plot(t, traj_below[:, 3], 'red', label='Persistence', linewidth=2)
axes[0, 0].set_xlabel('Time post infection')
axes[0, 0].set_ylabel('Cytokine pathology (P)')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_title('Cytokine Pathology Over Time')

# CD4 T cells
axes[0, 1].plot(t, traj_saddle[:, 2], color='black', label='At Saddle point', linewidth=2)
axes[0, 1].plot(t, traj_above[:, 2], 'blue', label='Clearance', linewidth=2)
axes[0, 1].plot(t, traj_below[:, 2], 'red', label='Persistence', linewidth=2)
axes[0, 1].set_xlabel('Time post infection')
axes[0, 1].set_ylabel('CD4 T cells (C)')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_title('CD4 T Cells Over Time')

# CD8 T cells
axes[1, 0].plot(t, traj_saddle[:, 1]/I_max, color='black', label='At Saddle point', linewidth=2)
axes[1, 0].plot(t, traj_above[:, 1]/I_max, 'blue', label='Clearance', linewidth=2)
axes[1, 0].plot(t, traj_below[:, 1]/I_max, 'red', label='Persistence', linewidth=2)
axes[1, 0].set_yscale('log')
axes[1, 0].set_xlabel('Time post infection')
axes[1, 0].set_ylabel('CD8 T cells (E/Imax)')
axes[1, 0].set_ylim([10**(-6), 1])
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_title('CD8 T Cells Over Time')

# Infected cells
axes[1, 1].plot(t, traj_saddle[:, 0]/I_max, color='black', label='At Saddle point', linewidth=2)
axes[1, 1].plot(t, traj_above[:, 0]/I_max, 'blue', label='Clearance', linewidth=2)
axes[1, 1].plot(t, traj_below[:, 0]/I_max, 'red', label='Persistence', linewidth=2)
axes[1, 1].set_yscale('log')
axes[1, 1].set_xlabel('Time post infection')
axes[1, 1].set_ylabel('Infected cells (I/Imax)')
axes[1, 1].set_ylim([10**(-6), 1])
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_title('Infected Cells Over Time')

plt.tight_layout()
plt.show()