import numpy as np
import matplotlib.pyplot as plt

file = "data/energies.txt"
M = np.loadtxt(file).transpose()

linestyles = ["-", "--"]
for i in range(0, len(M)):
    plt.plot(M[i], linestyle=linestyles[i % 2])

plt.ylabel(r"$\epsilon / t$")
plt.xlabel(r"$k / \pi$")
plt.tight_layout()
plt.show()