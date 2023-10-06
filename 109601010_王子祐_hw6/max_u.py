import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#df = pd.DataFrame(columns=["name","hpa-max","hpa-min","lat","max-lon","min-lon"])
df = pd.read_csv("meridian extreme.csv")
# 定義矩陣的維度
rows = 252
cols = 252
levels = 10
vars = 4
# 讀取二進制文件的數據
filename = 'fanapid.dat'  # 二進制文件的路徑
data = np.fromfile(filename, dtype='>f4')

# 重新組織數據為矩陣
data = data.reshape(vars,levels, rows, cols)
lat = np.linspace(4,43,252)
lon = np.linspace(95,145,252)
hgt1 = [1000,900,850,800,700,600,500,400,300,250]

lonx,hgt= np.meshgrid(lon,hgt1, indexing = 'xy')

# 提取各個變數的數據
u = np.array(data[0])
v = np.array(data[1])
w = np.array(data[2])
t = np.array(data[3])

def replace_above_threshold(matrix, threshold):
    mask = matrix >= threshold
    matrix[mask] = 0
    return matrix
threshold = 10**30
u = replace_above_threshold(u, threshold)
v = replace_above_threshold(v, threshold)
w = replace_above_threshold(w, threshold)
t = replace_above_threshold(t, threshold)

def find_closest_value(array, target):
    idx = np.abs(array - target).argmin()
    closest_value = array[idx]
    arg = np.argmin(abs(array-closest_value))
    return arg


target = 23.2
arg = find_closest_value(lat, target)

target = 115
arg_lon1 = find_closest_value(lon, target)

target = 130
arg_lon2 = find_closest_value(lon, target)

def find_extremum(matrix):
    # 將矩陣轉換為NumPy陣列
    arr = np.array(matrix)
    
    # 找到最大值的位置
    max_pos = np.unravel_index(np.argmax(arr), arr.shape)
    
    min_pos = np.unravel_index(np.argmin(arr), arr.shape)
    return max_pos,min_pos

max_point,min_point = find_extremum(t[:,arg,arg_lon1:arg_lon2])

df.loc[len(df)] = ["extreme-t",hgt[max_point[0],max_point[1]],hgt[min_point[0],min_point[1]]
                   ,lat[arg],lonx[max_point[0],max_point[1]+arg_lon1],lonx[min_point[0],min_point[1]+arg_lon1]]
level = np.linspace(-3,5,9)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
p = ax.contourf(lonx, hgt, w[:,arg,:],cmap="gist_ncar",levels=level)
plt.xlim(115, 130)
ax.set_xlabel("E")
ax.set_ylabel("hpa")
ax.set_ylim(ax.get_ylim()[::-1])
ax.scatter(lonx[max_point[0],max_point[1]+arg_lon1],hgt[max_point[0],max_point[1]],c ="red",label="max",s=200)
ax.scatter(lonx[min_point[0],min_point[1]+arg_lon1],hgt[min_point[0],min_point[1]],c ="blue",label="min",s=200)
plt.grid("--")
cb = plt.colorbar(p)
cb.set_label("vertical velocity")
plt.legend()
plt.title("meridian vertical velocity and max-t")
plt.savefig("meridian vertical velocity and max-t.jpg",dpi=400)
plt.show()
df.to_csv("meridian extreme.csv",index=False)