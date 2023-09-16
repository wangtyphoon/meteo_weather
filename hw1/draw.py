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
        
    def horizental_temparature_advection(self):
        t_adv = np.zeros([self.nlev,  self.nlat,self.mlon])
        for i in range(self.nlev):
            for j in range(self.nlat):
                for k in range(self.mlon):
                    if 1 <= j < self.nlat - 1 and 1 <= k < self.mlon - 1:
                        dx = self.dy * np.cos(self.lat[j] * np.pi / 180)
                        xvalue = self.u[i, j, k]*(self.t[i, j, k+1] - self.t[i, j, k-1]) / (2 * dx)
                        yvalue = self.v[i, j, k]*(self.t[i, j+1, k] - self.t[i, j-1, k]) / (2 * self.dy)
                        t_adv[i, j, k] = -xvalue -yvalue
                    else:
                        # 單邊插植
                        dx = self.dy * np.cos(self.lat[j] * np.pi / 180)
                        if k == 0:
                            xvalue = self.u[i, j, k]*(self.t[i, j, k+1] - self.t[i, j, k]) / dx
                        elif k == self.mlon - 1:
                            xvalue = self.u[i, j, k]*(self.t[i, j, k] - self.t[i, j, k-1]) / dx
                        else:
                            xvalue = (self.u[i, j, k + 1] - self.u[i, j, k - 1]) / (2 * dx)

                        if j == 0:
                            yvalue = self.v[i, j, k]*(self.t[i, j+1, k] - self.t[i, j, k]) / self.dy
                        elif j == self.nlat - 1:
                            yvalue = self.v[i, j, k]*(self.t[i, j, k] - self.t[i, j-1, k]) / self.dy
                        else:
                            yvalue = self.v[i, j, k]*(self.t[i, j+1, k] - self.t[i, j-1, k]) / (2 * self.dy)

                        t_adv[i, j, k] = xvalue + yvalue

        return t_adv

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

    def Relative_Vorticity(self):
        # 計算相對渦度
        rv = np.zeros([self.nlev,  self.nlat,self.mlon])
        # i:level j:lat m:lon
        for i in range(self.nlev):
            for j in range(self.nlat):
                for k in range(1, self.mlon):
                    if 1 <= j < self.nlat - 1 and 1 <= k < self.mlon - 1:
                        dx = self.dy * np.cos(self.lat[j] * np.pi/180)
                        xvalue = (self.v[i, j, k+1] - self.v[i, j, k-1]) / (2 * dx)
                        yvalue = (self.u[i, j+1, k] - self.u[i, j-1, k]) / (2 * self.dy)
                        rv[i, j, k] = xvalue - yvalue
                    else:
                        # 單邊插植
                        dx = self.dy * np.cos(self.lat[j] * np.pi / 180)
                        if k == 0:
                            xvalue = (self.v[i, j, k+1] - self.v[i, j, k]) /  dx
                        elif k == self.mlon - 1:
                            xvalue = (self.v[i, j, k] - self.v[i, j, k - 1]) / dx
                        else:
                            xvalue = (self.v[i, j, k + 1] - self.v[i, j, k - 1]) / (2 * dx)
                        
                        if j == 0:
                            yvalue = (self.u[i, j + 1, k] - self.u[i, j, k]) / self.dy
                        elif j == self.nlat - 1:
                            yvalue = (self.u[i, j, k] - self.u[i, j - 1, k]) / self.dy
                        else:
                            yvalue = (self.u[i, j + 1, k] - self.u[i, j - 1, k]) / (2 * self.dy)
                        
                        rv[i, j, k] = xvalue - yvalue
        return rv
    
    def plot_data(self, factor, title, label):
        # 繪製資料
        levels = [1000, 850, 700, 500, 300]
        os.makedirs(title[4:], exist_ok=True)
        for level in range(factor.shape[0]):
            plt.figure(figsize=(6, 3), dpi=400)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([90, 180, 15, 60], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND)  # 陸地特徵
            ax.add_feature(cfeature.COASTLINE)  # 海岸線特徵
            ax.add_feature(cfeature.BORDERS)  # 國界特徵
            var = factor[level, :, :]
            contour = ax.contourf(self.lon, self.lat, var, cmap='jet')  # 用色塊表示資料
            file = str(levels[level]) + title
            ax.set_title(file)
            ax.gridlines(draw_labels=[True, "x", "y", "bottom", "left"], linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

            cbar = plt.colorbar(contour, ax=ax, orientation='vertical', shrink=0.7, label=label)  # 色塊對應的色標
            plt.savefig(title[4:]+"/"+file+".png")
            plt.show()

if __name__ == "__main__":
    filename = 'output.bin'
    data_plotter = MyDataPlotter(filename)
    data_plotter.load_data()
    data_plotter.configure_parameters()
    t_adv = data_plotter.horizental_temparature_advection()
    div = data_plotter.Divergence()
    rv = data_plotter.Relative_Vorticity()
    data_plotter.plot_data(t_adv * 10000, 'hpa horizental Temparature advection', "(10^-4/s)")
    data_plotter.plot_data(div * 100000, 'hpa Divergence', "(10^-5/s)")
    data_plotter.plot_data(rv * 100000, 'hpa Relative Vorticity', "(10^-5/s)")  
