import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import laplace

a = 0.1
epsilon = 0.01
D_u = 5.0
D_v = 0.5

nx, ny = 100, 100
dx = dy = 1.0
dt = 0.01
steps = 10000
plot_interval = 100

u = np.random.rand(nx, ny) * 0.1
v = np.zeros((nx, ny))
u[nx//2 - 5:nx//2 + 5, ny//2 - 5:ny//2 + 5] = 1.0
frames = []

for step in range(steps):
    lap_u = laplace(u, mode='reflect')
    grad_vx, grad_vy = np.gradient(v, dx, dy)
    grad_v_mag = np.sqrt(grad_vx**2 + grad_vy**2)

    du = u * (u - a) * (1 - u) - v + D_u * lap_u
    dv = epsilon * u + D_v * grad_v_mag

    u += dt * du
    v += dt * dv

    if step % plot_interval == 0:
        frames.append(u.copy())

fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='hot', origin='lower')
ax.set_title("FitzHugh-Nagumo u(x, y, t)")
plt.colorbar(im, ax=ax)

def update(frame):
    im.set_data(frame)
    return [im]

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.show()
