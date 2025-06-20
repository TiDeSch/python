import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
k_n = 1*10**(-6)    # nucleation rate
k_a = 1*10**(-3)    # elongation rate
k_d = 5*10**(-3)    # dissociation rate
k_f = 2*10**(-3)    # fragmentation rate
n_c = 4   # critical nucleus size
m0 = 100.0  # initial monomer concentration

t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)
t_parts = np.max(t_span) / len(t_span)

def m(t, M):
    return m0 - M

def oosawa_model(t, y):
    P, M = y
    mt = m(t, M)
    dPdt = k_n * mt**n_c
    dMdt = n_c * k_n * mt**n_c - k_a * mt * P
    return [dPdt, dMdt]

# Oosawa + Dissociation
def oosawa_dissociation_model(t, y):
    P, M = y
    mt = m(t, M)
    dPdt = k_n * mt**n_c
    dMdt = n_c * k_n * mt**n_c + (k_a * mt - 2 * k_d) * P
    return [dPdt, dMdt]

# Oosawa + Fragmentation
def oosawa_fragmentation_model(t, y):
    P, M = y
    mt = m(t, M)
    dPdt = k_n * mt**n_c + k_f * (M - (2 * n_c + 1) * P)
    dMdt = n_c * k_n * mt**n_c + (2 * k_a * mt - n_c * (n_c - 1) * k_f) * P
    return [dPdt, dMdt]

y0 = [0.0, 0.0]
sol_oosawa = solve_ivp(oosawa_model, t_span, y0, t_eval=t_eval)
sol_dissociation = solve_ivp(oosawa_dissociation_model, t_span, y0, t_eval=t_eval)
sol_fragmentation = solve_ivp(oosawa_fragmentation_model, t_span, y0, t_eval=t_eval)

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
# Oosawa
ax.plot(sol_oosawa.t, m0 - sol_oosawa.y[1], label='m(t)')
ax.plot(sol_oosawa.t, sol_oosawa.y[0], label='P(t)')
ax.plot(sol_oosawa.t, sol_oosawa.y[1], label='M(t)')
ax.set_ylim([0,100])
ax.set_title("Oosawa Model")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("Time")
ax.set_ylabel("Concentration")

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
# Oosawa + Dissociation
axs[0].plot(sol_dissociation.t, m0 - sol_dissociation.y[1], label='m(t)')
axs[0].plot(sol_dissociation.t, sol_dissociation.y[0], label='P(t)')
axs[0].plot(sol_dissociation.t, sol_dissociation.y[1], label='M(t)')
axs[0].set_ylim([0,100])
axs[0].set_title("Oosawa + Dissociation")
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Concentration")

# Oosawa + Fragmentation
axs[1].plot(sol_fragmentation.t, m0 - sol_fragmentation.y[1], label='m(t)')
axs[1].plot(sol_fragmentation.t, sol_fragmentation.y[0], label='P(t)')
axs[1].plot(sol_fragmentation.t, sol_fragmentation.y[1], label='M(t)')
axs[1].set_ylim([0,100])
axs[1].set_title("Oosawa + Fragmentation")
axs[1].legend()
axs[1].grid(alpha=0.3)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Concentration")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
# Oosawa + Dissociation
ax.plot(sol_dissociation.t, m0 - sol_dissociation.y[1], label='m(t)')
ax.plot(sol_dissociation.t, sol_dissociation.y[0], label='P(t)')
ax.plot(sol_dissociation.t, sol_dissociation.y[1], label='M(t)')
ax.set_ylim([0,100])
ax.set_title("Oosawa + Dissociation")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("Time")
ax.set_ylabel("Concentration")

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
# Oosawa + Fragmentation
ax.plot(sol_fragmentation.t, m0 - sol_fragmentation.y[1], label='m(t)')
ax.plot(sol_fragmentation.t, sol_fragmentation.y[0], label='P(t)')
ax.plot(sol_fragmentation.t, sol_fragmentation.y[1], label='M(t)')
ax.set_ylim([0,100])
ax.set_title("Oosawa + Fragmentation")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlabel("Time")
ax.set_ylabel("Concentration")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
# P(t) Number of polymers
ax[0].plot(sol_oosawa.t, sol_oosawa.y[0], label='Oosawa')
ax[0].plot(sol_dissociation.t, sol_dissociation.y[0], label='+ Dissociation')
ax[0].plot(sol_fragmentation.t, sol_fragmentation.y[0], label='+ Fragmentation')
ax[0].set_ylim([0,30])
ax[0].set_title("P(t)")
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Concentration")

# M(t) Total mass of polymers
ax[1].plot(sol_oosawa.t, sol_oosawa.y[1], label='Oosawa')
ax[1].plot(sol_dissociation.t, sol_dissociation.y[1], label='+ Dissociation')
ax[1].plot(sol_fragmentation.t, sol_fragmentation.y[1], label='+ Fragmentation')
ax[1].set_ylim([0,100])
ax[1].set_title("M(t)")
ax[1].legend()
ax[1].grid(alpha=0.3)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Concentration")

plt.tight_layout()
plt.show()
