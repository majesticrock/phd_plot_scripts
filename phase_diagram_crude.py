import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import gzip

eps = 1e-5
if(len(sys.argv) > 1):
    data_folder = "data/" + sys.argv[1] + "/"
    name = sys.argv[1]
else:
    name = "T0.1"
    data_folder = f"data/phases/{name}/"

def pair_sort(pair_arr, sortBy):
    if len(pair_arr) < 2: 
        return
    n = len(pair_arr[sortBy])
    for i in range(sortBy, n):
        for j in range(i + 1, n):
            if pair_arr[sortBy][i] > pair_arr[sortBy][j]:
                pair_arr[sortBy][i], pair_arr[sortBy][j] = pair_arr[sortBy][j], pair_arr[sortBy][i]
                pair_arr[1 - sortBy][i], pair_arr[1 - sortBy][j] = pair_arr[1 - sortBy][j], pair_arr[1 - sortBy][i]

with gzip.open(data_folder + "cdw.dat.gz", 'rt') as f_open:
    CDW = np.loadtxt(f_open)
with gzip.open(data_folder + "afm.dat.gz", 'rt') as f_open:
    AFM = np.loadtxt(f_open)
with gzip.open(data_folder + "sc.dat.gz", 'rt') as f_open:
    SC =  abs(np.loadtxt(f_open))
with gzip.open(data_folder + "gamma_sc.dat.gz", 'rt') as f_open:
    GAMMA_SC = abs(np.loadtxt(f_open))
with gzip.open(data_folder + "xi_sc.dat.gz", 'rt') as f_open:
    XI_SC = abs(np.loadtxt(f_open))
with gzip.open(data_folder + "eta.dat.gz", 'rt') as f_open:
    ETA = abs(np.loadtxt(f_open))


with gzip.open(data_folder + "boundaries_cdw.dat.gz", 'rt') as f_open:
    BOUND_CDW = np.loadtxt(f_open)
    pair_sort(BOUND_CDW, 0)
with gzip.open(data_folder + "boundaries_afm.dat.gz", 'rt') as f_open:
    BOUND_AFM = np.loadtxt(f_open)
    pair_sort(BOUND_AFM, 0)
with gzip.open(data_folder + "boundaries_sc.dat.gz", 'rt') as f_open:
    BOUND_SC = np.loadtxt(f_open)
    pair_sort(BOUND_SC, 0)
with gzip.open(data_folder + "boundaries_gamma_sc.dat.gz", 'rt') as f_open:
    BOUND_GAMMA_SC = np.loadtxt(f_open)
    pair_sort(BOUND_GAMMA_SC, 0)
with gzip.open(data_folder + "boundaries_xi_sc.dat.gz", 'rt') as f_open:
    BOUND_XI_SC = np.loadtxt(f_open)
    pair_sort(BOUND_XI_SC, 1)
with gzip.open(data_folder + "boundaries_eta.dat.gz", 'rt') as f_open:
    BOUND_ETA = np.loadtxt(f_open)
    pair_sort(BOUND_ETA, 0)


labels = ["T", "U"]
T_SIZE = len(CDW)
U_SIZE = len(CDW[0])

with gzip.open(data_folder + "cdw.dat.gz", 'rt') as fp:
    for i, line in enumerate(fp):
        if i == 2:
            ls = line.split()
            labels[0] = ls[1].split("_")[0]
            U = np.linspace(float(ls[1].split("=")[1]), float(ls[2].split("=")[1]), U_SIZE+1)[:U_SIZE]
        elif i == 3:
            ls = line.split()
            labels[1] = ls[1].split("_")[0]
            T = np.linspace(float(ls[1].split("=")[1]), float(ls[2].split("=")[1]), T_SIZE+1)[:T_SIZE]
        elif i > 3:
            break

for i in range(0, T_SIZE):
    for j in range(0, U_SIZE):
        if(CDW[i][j] > eps):
            CDW[i][j] = 1
        else:
            CDW[i][j] = 0

        if(AFM[i][j] > eps):
            AFM[i][j] = 1
        else:
            AFM[i][j] = 0

        if(SC[i][j] > eps):
            SC[i][j] = 1
        else:
            SC[i][j] = 0

        if(GAMMA_SC[i][j] > eps):
            GAMMA_SC[i][j] = 1
        else:
            GAMMA_SC[i][j] = 0

        if(XI_SC[i][j] > eps):
            XI_SC[i][j] = 1
        else:
            XI_SC[i][j] = 0

        if(ETA[i][j] > eps):
            ETA[i][j] = 1
        else:
            ETA[i][j] = 0


X, Y = np.meshgrid(U, T)
cmap0 = colors.ListedColormap([colors.to_rgba('white', 0), colors.to_rgba('C0', 0.5)])
cmap1 = colors.ListedColormap([colors.to_rgba('white', 0), colors.to_rgba('C1', 0.5)])
cmap2 = colors.ListedColormap([colors.to_rgba('white', 0), colors.to_rgba('C2', 0.5)])
cmap3 = colors.ListedColormap([colors.to_rgba('white', 0), colors.to_rgba('C3', 0.5)])
cmap4 = colors.ListedColormap([colors.to_rgba('white', 0), colors.to_rgba('C4', 0.5)])

fig, ax = plt.subplots()

cset0 = ax.contourf(X, Y, SC, 1, cmap=cmap0)
cset1 = ax.contourf(X, Y, CDW, 1, cmap=cmap1)
cset2 = ax.contourf(X, Y, AFM, 1, cmap=cmap2)
cset3 = ax.contourf(X, Y, XI_SC, 1, cmap=cmap3)
#cset4 = ax.contourf(X, Y, GAMMA_SC, 1, cmap=cmap4)
#cbar = fig.colorbar(cset1)

from matplotlib.patches import Patch

legend_elements = [Patch(facecolor='C0', label=r'$s$'),
            Patch(facecolor='C1', label=r'CDW'),
            Patch(facecolor='C2', label=r'AFM'),
            Patch(facecolor='C3', label=r'$d_{x^2 - y^2}$')]
            #,Patch(facecolor='C4', label=r'$\tilde{s}$')]
ax.legend(handles=legend_elements, loc='upper left')

if(len(BOUND_CDW) == 2):
    ax.plot(BOUND_CDW[1], BOUND_CDW[0], "k.")
if(len(BOUND_AFM) == 2):
    ax.plot(BOUND_AFM[1], BOUND_AFM[0], "k.")
if(len(BOUND_SC) == 2):
    ax.plot(BOUND_SC[1], BOUND_SC[0], "k.")
if(len(BOUND_GAMMA_SC) == 2):
    ax.plot(BOUND_GAMMA_SC[1], BOUND_GAMMA_SC[0], "k.")
if(len(BOUND_XI_SC) == 2):
    ax.plot(BOUND_XI_SC[1], BOUND_XI_SC[0], "k.")
if(len(BOUND_ETA) == 2):
    ax.plot(BOUND_ETA[1], BOUND_ETA[0], "k.")

plt.xlabel(r"$" + labels[0] + "/t$")
plt.ylabel(r"$" + labels[1] + "/t$")

import os
if not os.path.exists("python/build"):
    os.makedirs("python/build")
plt.savefig(f"python/build/{os.path.basename(__file__).split('.')[0]}_{name}.pdf")
plt.show()