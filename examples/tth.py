import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots

    plt.style.use(["science", "grid"])
except ImportError:
    pass
plt.rcParams["figure.figsize"] = 16, 9

N_GRID = 50

data = np.load("pdf_grid.npy")
plt.contourf(
    np.reshape(data.T[0], shape=(N_GRID, N_GRID)),
    np.reshape(data.T[1], shape=(N_GRID, N_GRID)),
    np.reshape(np.log10(data.T[2]), shape=(N_GRID, N_GRID)),
    levels=20,
)
plt.colorbar()
plt.xlabel(r"$x_3$")
plt.ylabel(r"$x_4$")
plt.show()
