import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

eps = 1e-8

CDW = (np.loadtxt("data/basic_hubbard_cdw.txt"))
SC =  (np.loadtxt("data/basic_hubbard_sc.txt"))
ETA = (np.loadtxt("data/basic_hubbard_eta.txt"))

T_SIZE = len(CDW)
U_SIZE = len(CDW[0])

with open("data/basic_hubbard_cdw.txt") as fp:
    for i, line in enumerate(fp):
        if i == 2:
            ls = line.split()
            U = np.linspace(float(ls[1].split("=")[1]), float(ls[2].split("=")[1]), U_SIZE)
        elif i == 3:
            ls = line.split()
            T = np.linspace(float(ls[1].split("=")[1]), float(ls[2].split("=")[1]), T_SIZE)
        elif i > 3:
            break


for i in range(0, T_SIZE):
    for j in range(0, U_SIZE):
        if(CDW[i][j] > eps):
            CDW[i][j] = 1
        else:
            CDW[i][j] = 0
        if(SC[i][j] > eps):
            SC[i][j] = 1
        else:
            SC[i][j] = 0
        if(ETA[i][j] > eps):
            ETA[i][j] = 1
        else:
            ETA[i][j] = 0

X, Y = np.meshgrid(U, T)
cmap1 = colors.ListedColormap(['white', 'green'])
cmap2 = colors.ListedColormap(['white', 'red'])

fig, ax = plt.subplots()

mpl.rcParams["hatch.linewidth"] = 2.5
cset1 = ax.contourf(X-0.5*(U[1]-U[0]), Y-0.5*(T[1]-T[0]), SC, 1, cmap=cmap1, hatches=[None, None])
cset2 = ax.contourf(X-0.5*(U[1]-U[0]), Y-0.5*(T[1]-T[0]), CDW, 1, cmap=cmap2, hatches=[None, r"//"], alpha=0.4)
cset3 = ax.contourf(X-0.5*(U[1]-U[0]), Y-0.5*(T[1]-T[0]), ETA, 1, cmap=cmap2, hatches=[None, r"\\"], alpha=0)
#cbar = fig.colorbar(cset1)

from matplotlib.patches import Patch

legend_elements = [Patch(facecolor='green', label=r'$\Delta_{SC}$'),
            Patch(facecolor='red', label=r'$\Delta_{CDW}$')]
ax.legend(handles=legend_elements, loc='upper right')

plt.xlabel(r"$U / t$")
plt.ylabel(r"$T / t$")
plt.show()