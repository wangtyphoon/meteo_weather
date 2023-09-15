import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd

df = pd.DataFrame(columns=['text', 'max_lon', 'max_lat', 'min_lon',"min_lat"])
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
lonx,laty= np.meshgrid(lon,lat, indexing = 'xy')
hgt = [1000,900,850,800,700,600,500,400,300,250]
h = [2,4,8]
# 提取各個變數的數據
u = np.array(data[0])
v = np.array(data[1])
w = np.array(data[2])
t = np.array(data[3])

def replace_above_threshold(matrix, threshold):
    mask = matrix > threshold
    matrix[mask] = 0
    return matrix
threshold = 10**10
u = replace_above_threshold(u, threshold)
v = replace_above_threshold(v, threshold)
w = replace_above_threshold(w, threshold)
t = replace_above_threshold(t, threshold)



# 打印各個變數的矩陣形狀
print("U shape:", u.shape)
print("V shape:", v.shape)
print("W shape:", w.shape)
print("T shape:", t.shape)

def find_extremum(matrix):
    # 將矩陣轉換為NumPy陣列
    arr = np.array(matrix)
    
    # 找到最大值的位置
    max_pos = np.unravel_index(np.argmax(arr), arr.shape)
    
    min_pos = np.unravel_index(np.argmin(arr), arr.shape)
    return max_pos,min_pos

def find_closest_value(array, target):
    idx = np.abs(array - target).argmin()
    closest_value = array[idx]
    arg = np.argmin(abs(array-closest_value))
    return arg

target = 18
lat_arg1 = find_closest_value(lat, target)
target = 28
lat_arg2 = find_closest_value(lat, target)

target = 118
lon_arg1 = find_closest_value(lon, target)
target = 128
lon_arg2 = find_closest_value(lon, target)


for i in range(2,10):
    max_point,min_point = find_extremum(u[i,lat_arg1:lat_arg2,lon_arg1:lon_arg2])
    df.loc[len(df)] = [str(hgt[i])+"u",lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1],
                        lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1]]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([118, 128, 18, 28], crs=ccrs.PlateCarree())
    level = np.linspace(0,65,14)
    ax.coastlines()
    ax.set_xticks(np.linspace(118,128,11), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(18,28,11), crs=ccrs.PlateCarree())
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    z = (u[i,:,:]**2+v[i,:,:]**2)**0.5
    p = ax.contourf(lonx, laty, z, level,transform=ccrs.PlateCarree(),cmap="gist_ncar")
    plt.scatter(lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1], color='red', label='Max',s=200)
    plt.scatter(lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1], color='blue', label='Min',s=200)
    plt.colorbar(p)
    plt.grid("--")
    plt.legend()
    plt.title(str(hgt[i])+"max-u")
    plt.savefig(str(hgt[i])+"max-u.jpg",dpi=400)
    plt.show()

for i in range(2,10):
    max_point,min_point = find_extremum(v[i,lat_arg1:lat_arg2,lon_arg1:lon_arg2])
    df.loc[len(df)] = [str(hgt[i])+"v",lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1],
                        lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1]]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([118, 128, 18, 28], crs=ccrs.PlateCarree())
    level = np.linspace(0,65,14)
    ax.coastlines()
    ax.set_xticks(np.linspace(118,128,11), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(18,28,11), crs=ccrs.PlateCarree())
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    z = (u[i,:,:]**2+v[i,:,:]**2)**0.5
    p = ax.contourf(lonx, laty, z, level,transform=ccrs.PlateCarree(),cmap="gist_ncar")
    plt.scatter(lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1], color='red', label='Max',s=200)
    plt.scatter(lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1], color='blue', label='Min',s=200)
    plt.colorbar(p)
    plt.grid("--")
    plt.legend()
    plt.title(str(hgt[i])+"max-v")
    plt.savefig(str(hgt[i])+"max-v.jpg",dpi=400)
    plt.show()


for i in range(2,10):
    max_point,min_point = find_extremum(w[i,lat_arg1:lat_arg2,lon_arg1:lon_arg2])
    df.loc[len(df)] = [str(hgt[i])+"t",lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1],
                        lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1]]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([118, 128, 18, 28], crs=ccrs.PlateCarree())
    level = np.linspace(0,65,14)
    ax.coastlines()
    ax.set_xticks(np.linspace(118,128,11), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(18,28,11), crs=ccrs.PlateCarree())
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    z = (u[i,:,:]**2+v[i,:,:]**2)**0.5
    p = ax.contourf(lonx, laty, z, level,transform=ccrs.PlateCarree(),cmap="gist_ncar")
    plt.scatter(lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1], color='red', label='Max',s=200)
    plt.scatter(lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1], color='blue', label='Min',s=200)
    plt.colorbar(p)
    plt.grid("--")
    plt.legend()
    plt.title(str(hgt[i])+"max-w")
    plt.savefig(str(hgt[i])+"max-w.jpg",dpi=400)
    plt.show()


for i in range(2,10):
    max_point,min_point = find_extremum(t[i,lat_arg1:lat_arg2,lon_arg1:lon_arg2])
    df.loc[len(df)] = [str(hgt[i])+"w",lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1],
                        lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1]]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([118, 128, 18, 28], crs=ccrs.PlateCarree())
    level = np.linspace(0,65,14)
    ax.coastlines()
    ax.set_xticks(np.linspace(118,128,11), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(18,28,11), crs=ccrs.PlateCarree())
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    z = (u[i,:,:]**2+v[i,:,:]**2)**0.5
    p = ax.contourf(lonx, laty, z, level,transform=ccrs.PlateCarree(),cmap="gist_ncar")
    plt.scatter(lonx[max_point[0]+lat_arg1,max_point[1]+lon_arg1],laty[max_point[0]+lat_arg1,max_point[1]+lon_arg1], color='red', label='Max',s=200)
    plt.scatter(lonx[min_point[0]+lat_arg1,min_point[1]+lon_arg1],laty[min_point[0]+lat_arg1,min_point[1]+lon_arg1], color='blue', label='Min',s=200)
    plt.colorbar(p)
    plt.grid("--")
    plt.legend()
    plt.title(str(hgt[i])+"max-t")
    plt.savefig(str(hgt[i])+"max-t.jpg",dpi=400)
    plt.show()

df.to_csv('output.csv', index=False)
