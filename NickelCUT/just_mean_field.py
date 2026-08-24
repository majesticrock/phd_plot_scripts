import numpy as np

L = 40
N = L*L

U = -1.

k = np.linspace(-np.pi, np.pi, L, endpoint=False)
X, Y = np.meshgrid(k, k)

dispersion = -2. * (np.cos(X) + np.cos(Y))

Delta = 0.1
Delta_new = 0.0

error = 100.

while error > 1e-8:
    Delta_new = (-0.5 * U / N) * np.sum( Delta / np.sqrt(dispersion.flatten()**2 + Delta**2) )
    error = np.abs(Delta_new - Delta)
    Delta = Delta_new
    
    print(f"Error = {error},  Delta_max = {np.max(np.abs(Delta))}")
    
print("Selfconsistency converged; Delta =", Delta)