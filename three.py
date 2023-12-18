import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

def three_body_equations(y, t, G, m1, m2, m3):
    x1, x2, x3, y1, y2, y3, z1, z2, z3, vx1, vx2, vx3, vy1, vy2, vy3, vz1, vz2, vz3 = y

    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)

    dydt = [
        vx1, vx2, vx3,
        vy1, vy2, vy3,
        vz1, vz2, vz3,
        G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3,
        G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3,
        G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3,
        G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3,
        G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3,
        G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3,
        G * m2 * (z2 - z1) / r12**3 + G * m3 * (z3 - z1) / r13**3,
        G * m1 * (z1 - z2) / r12**3 + G * m3 * (z3 - z2) / r23**3,
        G * m1 * (z1 - z3) / r13**3 + G * m2 * (z2 - z3) / r23**3,
    ]

    return dydt

# 初始条件
y0 = [
    0.15, -0.35, 0.0,
    0.1, 1.0, 0.1,
    0.3, 0.0, .2,
    0.4662036850, 0.4323657300, -0.8981246360,
    0.9324073490, 0.8647314640, 0.5238853240,
    0.0, 0.0, 0.0,  # 添加三个初始速度的值
]

# 时间点
t = np.linspace(0, 20, 1000000)

# 参数
G = 1.0
m1 = 1.0
m2 = 2.0
m3 = 1.0

# 求解微分方程
solution = odeint(three_body_equations, y0, t, args=(G, m1, m2, m3))

# 提取坐标
x1, x2, x3, y1, y2, y3, z1, z2, z3, _, _, _, _, _, _, _, _, _ = solution.T

# 可视化轨迹
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, y1, z1, label='Mass 1')
ax.plot(x2, y2, z2, label='Mass 2')
ax.plot(x3, y3, z3, label='Mass 3')
ax.set_title('Three-Body Simulation in 3D')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.show()
