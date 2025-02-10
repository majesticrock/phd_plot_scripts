import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

class HeatmapPlotter:
    def __init__(self, data_frame, parameter_name, xlabel=r'$k / k_\mathrm{F}$', ylabel='Y-axis', zlabel=r'$\gamma$'):
        for i, row in data_frame.iterrows():
            print(row["g"])
        print("\n\n\n#######################")
        df = data_frame.query(f"{parameter_name} > 8 & {parameter_name} < 25").sort_values(parameter_name)
        data_size = data_frame["inner_discretization"].iloc[0] // 2
        additional_skip = int(0.75 * data_size)
        n_skip = 4
        self.x = (df["data"].iloc[0]["ks"] / df["k_F"].iloc[0]).to_numpy()[data_size+additional_skip:3*data_size-additional_skip:n_skip]
        self.y = (df[parameter_name]).to_numpy()
        self.gaps = np.array([ (row['Delta_Coulomb'] + row['Delta_Phonon'])[data_size+additional_skip:3*data_size-additional_skip:n_skip] for row in df['data'] ])
        self.eps  = np.array([ (1e3 * row['xis']     + row['Delta_Fock']  )[data_size+additional_skip:3*data_size-additional_skip:n_skip] for row in df['data'] ])
        self.z = np.sqrt(self.gaps**2 + self.eps**2)
        for i in range(len(df)):
            self.z[i] = 1 - self.z[i] / df["Delta_max"].iloc[i]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.fig, self.ax = plt.subplots()
        
        self.plot()

    def plot(self, cmap='magma'):
        #divnorm=colors.TwoSlopeNorm(vmin=self.z.min(), vcenter=0., vmax=-self.z.min())
        norm = colors.PowerNorm(gamma=0.33, vmin=0, vmax=self.z.max())
        contour = self.ax.pcolormesh(self.x, self.y, self.z, cmap=cmap, norm=norm,
                                     shading="gouraud", 
                                     rasterized=True, 
                                     edgecolors="face")
        
        cbar = self.fig.colorbar(contour, ax=self.ax, extend='max')
        cbar.set_label(self.zlabel)

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def customize_ticks(self, x_ticks=None, y_ticks=None):
        if x_ticks is not None:
            self.ax.set_xticks(x_ticks)
        if y_ticks is not None:
            self.ax.set_yticks(y_ticks)

    def show_plot(self):
        plt.show()

    def save_plot(self, filename):
        self.fig.savefig(filename)


import __path_appender as __ap
__ap.append()
from get_data import *

all_data = load_pickle("continuum/offset_25/", "gaps.pkl").query("discretization == 30000 & T == 0")

g_small_screening = HeatmapPlotter(all_data.query("coulomb_scaling == 1 & lambda_screening == 0.0001 & k_F == 4.25 & omega_D == 10"), "Delta_max", ylabel=r"$\Delta_{max}$ $(\mathrm{meV})$")
g_large_screening = HeatmapPlotter(all_data.query("coulomb_scaling == 1 & lambda_screening == 1      & k_F == 4.25 & omega_D == 10"), "Delta_max", ylabel=r"$\Delta_{max}$ $(\mathrm{meV})$")
g_no_coulomb      = HeatmapPlotter(all_data.query("coulomb_scaling == 0 & lambda_screening == 0      & k_F == 4.25 & omega_D == 10"), "Delta_max", ylabel=r"$\Delta_{max}$ $(\mathrm{meV})$")


plt.show()