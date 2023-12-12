import matplotlib.pyplot as plt
import numpy as np


def lorenz(xyzth, *, s=10, r=24.74, b=2.6666667,f=2.5):
    
    x, y, z,theta = xyzth
    theta = 1
    x_dot = s*(y - x) + f*np.cos(theta/180*np.pi)
    y_dot = r*x - y - x*z + f*np.sin(theta/180*np.pi)
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot,theta])


dt = 0.01
num_steps = 5000
theta = 45

xyzs = np.empty((num_steps + 1, 4))  # Need one more for the initial values
xyzs[0] = (0., 10., 0.,theta)  # Set initial values
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt
    
    

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*xyzs.T[0:3], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

# Create an array of time values
time_values = dt * np.arange(num_steps + 1)

# Plot x-t plot
plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(time_values, xyzs[:, 0], lw=0.5)
plt.grid("--")
plt.xlabel("Time (t)")
plt.ylabel("X Axis")
plt.title("X-t Plot")

# Plot x-z plot
plt.subplot(3, 2, 2)
plt.plot(xyzs[:, 2], xyzs[:, 0], lw=0.5)
plt.grid("--")
plt.xlabel("Z Axis")
plt.ylabel("X Axis")
plt.title("X-z Plot")

# Plot z-t plot
plt.subplot(3, 2, 3)
plt.plot(time_values, xyzs[:, 2], lw=0.5)
plt.grid("--")
plt.xlabel("Time (t)")
plt.ylabel("Z Axis")
plt.title("Z-t Plot")

# Plot z-y plot
plt.subplot(3, 2, 4)
plt.plot(xyzs[:, 1], xyzs[:, 2], lw=0.5)
plt.grid("--")
plt.xlabel("Y Axis")
plt.ylabel("Z Axis")
plt.title("Z-y Plot")

# Plot y-t plot
plt.subplot(3, 2, 5)
plt.plot(time_values, xyzs[:, 1], lw=0.5)
plt.grid("--")
plt.xlabel("Time (t)")
plt.ylabel("Y Axis")
plt.title("Y-t Plot")

# Plot x-y plot
plt.subplot(3, 2, 6)
plt.plot(xyzs[:, 1], xyzs[:, 0], lw=0.5)
plt.grid("--")
plt.xlabel("Y Axis")
plt.ylabel("X Axis")
plt.title("X-y Plot")

plt.tight_layout()
plt.show()