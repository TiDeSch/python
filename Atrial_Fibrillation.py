import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 200               # Grid size
τ = 50                # Refractory period
T = 220               # Pacemaker period
ν = 0.15              # Fraction of transverse connections
δ = 0.10              # Fraction of dysfunctional cells
ε = 0.20              # Probability of failed excitation
steps = 1000          # Number of simulation steps

state = np.zeros((L, L), dtype=int)  # 0=resting, 1=excited, 2+=refractory
next_state = state.copy()

transverse_conn = np.random.rand(L, L-1) < ν
dysfunctional = np.random.rand(L, L) < δ


frames = []
pacemaker_timer = 0

def excite_neighbors(i, j, state, next_state):
    neighbors = []
    if i > 0:
        neighbors.append((i-1, j))  # longitudinal up
    if i < L-1:
        neighbors.append((i+1, j))  # longitudinal down
    if j > 0 and transverse_conn[i, j-1]:
        neighbors.append((i, j-1))  # transverse left
    if j < L-1 and transverse_conn[i, j]:
        neighbors.append((i, j+1))  # transverse right

    for ni, nj in neighbors:
        if state[ni, nj] == 0:  # resting
            if not dysfunctional[ni, nj] or np.random.rand() > ε:
                next_state[ni, nj] = 1  # excite

for t in range(steps):
    next_state[:] = state

    for i in range(L):
        for j in range(L):
            if state[i, j] == 1:  # excited
                next_state[i, j] = 2
                excite_neighbors(i, j, state, next_state)
            elif state[i, j] >= 2:
                next_state[i, j] = state[i, j] + 1 if state[i, j] < τ+1 else 0

    if pacemaker_timer == 0:
        for i in range(L):
            if not dysfunctional[i, 0] or np.random.rand() > ε:
                next_state[i, 0] = 1
    pacemaker_timer = (pacemaker_timer + 1) % T

    state[:] = next_state
    frames.append(state.copy())

fig = plt.figure(figsize=(6, 6))
ims = [[plt.imshow(frame, cmap='hot', animated=True)] for frame in frames[::10]]
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
plt.show()
