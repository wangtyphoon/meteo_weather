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
        self.dy = 6378000 * 1.875 * np.pi/180
        

    def load_data(self):
        self.data = np.fromfile(self.filename, dtype='<f4')
        self.data = self.data.reshape(self.var, self.nlev,  self.nlat,self.mlon)

    def configure_parameters(self):
        
        self.lon = np.linspace(90, 180, self.mlon)
        self.lat = np.linspace(15, 60, self.nlat)
        self.h = self.data[0, :, :, :]
        self.u = self.data[1, :, :, :]
        self.v = self.data[2, :, :, :]
        self.t = self.data[3, :, :, :]
    
    def horizen_temparature_advection(self):
        t_adv = np.zeros([self.nlev,  self.nlat,self.mlon])
        # i:level j:lat m:lon
        for i in range(self.nlev):
            for j in range(1,self.nlat-1):
                dx = self.dy*np.cos(self.lat[j]*np.pi/180)
                for k in range(1,self.mlon-1):
                    xvalue = (self.v[i,j,k+1]-self.v[i,j,k-1])/(2*dx)
                    yvalue = (self.u[i,j+1,k]-self.u[i,j-1,k])/(2*self.dy)
                    t_adv[i,j,k] = xvalue-yvalue
        return t_adv
    
    def plot_data(self,factor,title,label):                
        levels = [1000,850,700,500,300]
        for level in range(factor.shape[0]):
            plt.figure(figsize=(6, 3), dpi=400)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([90, 180, 15, 60], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            var = factor[level,:,:]
            contour = ax.contourf(self.lon, self.lat, var,cmap = 'jet')
            ax.set_title(str(levels[level])+title)
            ax.gridlines(draw_labels=[True,"x", "y", "bottom", "left"], linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

            cbar = plt.colorbar(contour, ax=ax, orientation='vertical', shrink=0.7, label=label)
            plt.show()

if __name__ == "__main__":
    filename = 'output.bin'
    data_plotter = MyDataPlotter(filename)
    data_plotter.load_data()
    data_plotter.configure_parameters()
    t_adv = data_plotter.horizen_temparature_advection()
    data_plotter.plot_data(t_adv*100000,'hpa Relative Vortivity Field',"Temparature (10-5e K)")
