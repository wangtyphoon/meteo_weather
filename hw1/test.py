import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

class MyDataPlotter:
    def __init__(self, filename):
        self.filename = filename
        self.nlat = 25
        self.mlon = 49
        self.nlev = 5
        self.var = 4
        self.load_data()
        self.configure_parameters()
        self.horizen_temparature_advection()
        # self.plot_data()

    def load_data(self):
        self.data = np.fromfile(self.filename, dtype='<f4')
        self.data = self.data.reshape(self.var, self.nlev, self.nlat, self.mlon)

    def configure_parameters(self):
        
        self.lon = np.linspace(90, 180, self.mlon)
        self.lat = np.linspace(15, 60, self.nlat)
        self.h = self.data[0, :, :, :]
        self.u = self.data[1, :, :, :]
        self.v = self.data[2, :, :, :]
        self.t = self.data[3, :, :, :]
    
    def horizen_temparature_advection(self):
        for i in range(self.nlev):
            for j in range(1,self.nlat-1):
                for k in range(1,self.mlon-1):
                    print(i,j)

    def plot_data(self):
        plt.figure(figsize=(6, 3), dpi=400)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([90, 180, 15, 60], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.contourf(self.lon, self.lat, self.t[0, :, :])
        plt.show()

if __name__ == "__main__":
    filename = 'output.bin'
    data_plotter = MyDataPlotter(filename)
