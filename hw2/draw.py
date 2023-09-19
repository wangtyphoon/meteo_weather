import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import os
class MyDataPlotter:
    def __init__(self, filename):
        self.filename = filename
        self.nlat = 25  # 緯度格點數
        self.mlon = 49  # 經度格點數
        self.nlev = 5   # 垂直層數
        self.var = 4    # 變數數量
        self.dy = 6378000 * 1.875 * np.pi/180  # 經度間距對應的米數
        self.omega = 7.29*100000  # 地球自轉速率

    def load_data(self):
        # 載入資料
        self.data = np.fromfile(self.filename, dtype='<f4')
        self.data = self.data.reshape(self.var, self.nlev,  self.nlat, self.mlon)

    def configure_parameters(self):
        # 設定參數
        self.lon = np.linspace(90, 180, self.mlon)  # 經度範圍
        self.lat = np.linspace(15, 60, self.nlat)   # 緯度範圍
        self.h = self.data[0, :, :, :]  # 高度場
        self.u = self.data[1, :, :, :]  # 東向風場
        self.v = self.data[2, :, :, :]  # 北向風場
        self.t = self.data[3, :, :, :]  # 溫度場

    def Divergence(self):
        # 計算相對渦度
        div = np.zeros([self.nlev,  self.nlat,self.mlon])
        # i:level j:lat m:lon
        
        for i in range(self.nlev):
            for j in range(self.nlat):
                for k in range(self.mlon):
                    if 1 <= j < self.nlat - 1 and 1 <= k < self.mlon - 1:
                        dx = self.dy * np.cos(self.lat[j] * np.pi / 180)
                        xvalue = (self.u[i, j, k + 1] - self.u[i, j, k - 1]) / (2 * dx)
                        yvalue = (self.v[i, j + 1, k] - self.v[i, j - 1, k]) / (2 * self.dy)
                        div[i, j, k] = xvalue + yvalue
                    else:
                        # 單邊插植
                        dx = self.dy * np.cos(self.lat[j] * np.pi / 180)
                        if k == 0:
                            xvalue = (self.u[i, j, k + 1] - self.u[i, j, k]) / dx
                        elif k == self.mlon - 1:
                            xvalue = (self.u[i, j, k] - self.u[i, j, k - 1]) / dx
                        else:
                            xvalue = (self.u[i, j, k + 1] - self.u[i, j, k - 1]) / (2 * dx)

                        if j == 0:
                            yvalue = (self.v[i, j + 1, k] - self.v[i, j, k]) / self.dy
                        elif j == self.nlat - 1:
                            yvalue = (self.v[i, j, k] - self.v[i, j - 1, k]) / self.dy
                        else:
                            yvalue = (self.v[i, j + 1, k] - self.v[i, j - 1, k]) / (2 * self.dy)

                        div[i, j, k] = xvalue + yvalue

        return div
    
    def Vertical_Speed(self,div):
        vs = np.zeros([self.nlev,  self.nlat,self.mlon])
        p = [85,150,175,200,300]
        #initial vertical speed
        for i in range(self.nlev):
            for j in range(self.nlat):
                for k in range(self.mlon):
                    if i == 0:
                        vs[i, j, k] = div[i, j, k]*p[i]
                    elif i > 0:
                        vs[i, j, k] = vs[i-1, j, k]+div[i, j, k]*p[i]
        # #correction error
        # # 創建一個形狀為 [5, 25, 49] 的全零的 3D 數組
        expanded_error = np.zeros([5, 25, 49])
        for i in range(self.nlev):
            for j in range(self.nlat):
                for k in range(self.mlon):
                    expanded_error[i,j,k] = vs[4,j,k]/910
        div_new = div - expanded_error
        # # correction div
        vs_new =  np.zeros([self.nlev,  self.nlat,self.mlon], dtype=float)
        for i in range(self.nlev):
            for j in range(self.nlat):
                for k in range(self.mlon):                   
                    if i == 0:
                        vs_new [i, j, k] = div_new[i, j, k]*p[i]
                    elif i > 0:
                        vs_new [i, j, k] = vs_new[i-1, j, k]+div_new[i, j, k]*p[i]
        np.save('w_new.npy', vs_new)

        return vs_new
    
    def plot_data(self, factor, title, label):
        # 繪製資料
        level = [1010, 925, 775, 600, 400,100]
        # os.makedirs(title[5:], exist_ok=True)
        plt.figure(figsize=(6, 3), dpi=400)
        var = np.zeros((6,25))
        var[1:,:] = factor[:, :, 16]
        contour = plt.contour(self.lat, level, var, cmap='jet',levels = np.linspace(-0.002,0.002,9))  # 用色塊表示資料
        plt.title(title)
        plt.xticks(np.linspace(15,60,10))
        plt.yscale('log')  # 設置 y 軸為對數尺度
        plt.yticks(np.linspace(1000,100,10))
        # 設置 y 軸刻度標籤格式為指數形式
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        cbar = plt.colorbar(contour, orientation='vertical', shrink=0.7, label=label)  # 色塊對應的色標
        plt.gca().invert_yaxis()  # 倒轉 y 軸
        plt.ylim(1010, 100)  # 設置 y 軸範圍
        # 添加格線
        plt.grid(True, linestyle='--', alpha=0.5)
        # plt.savefig(title[5:] + "/" + title + ".png")
        plt.show()


if __name__ == "__main__":
    filename = 'output.bin'
    data_plotter = MyDataPlotter(filename)
    data_plotter.load_data()
    data_plotter.configure_parameters()
    div = data_plotter.Divergence()
    vs = data_plotter.Vertical_Speed(div)
    data_plotter.plot_data(vs, "120E vetical velocity", "m/s)")
