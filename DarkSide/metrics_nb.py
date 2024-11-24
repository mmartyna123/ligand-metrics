# %%
from blobify import *

map, scale, origin = load_map("6HCY.mrc.gz")
blob = load_blob("ours_6HCY_C_502_HEM.npz")
ligand = load_ligands("6HCY.cif")["6HCY_C_502_HEM"]
offset, size = get_offset_and_size(ligand)
atoms = ligand_to_atoms(ligand, offset, scale, origin)
len(atoms)

# %%
def create_ligand(shape, atoms, min=0, max=1):
    """
    Create a complex 3D blob using multiple Gaussian functions.

    Parameters:
    shape (tuple): Dimensions of the voxel grid (x, y, z).
    atoms (list of atoms): List of centers and sigmas for the Gaussian blobs.
    min (float): Lowest density value.
    max (float): Highest density value.

    Returns:
    np.ndarray: 3D array representing the voxel grid with blob values.
    """
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    z = np.linspace(0, shape[2] - 1, shape[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    blob = np.zeros(shape)
    for center, sigma in atoms:
        blob += np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2) / (2 * sigma.mean() ** 2))

    blob -= blob.min()
    blob /= blob.max()

    blob *= (max - min)
    blob += min

    return blob

gaussian = create_ligand(blob.shape, atoms, blob[blob > map.min()].min(), blob.max())
plot_blob_comparison(gaussian, blob)
plt.show()

# %%
import sys
sys.path.append("../LightSide")

from metrics import *

# %%
def compare_deferred(metrics):
    names = []
    elements = []

    def append(name, ab):
        names.append(name)
        elements.append(ab)
        return [metric(*ab) for name, metric in metrics]

    def table():
        longest_metric_name_len = max([len(name) for name, _ in metrics])
        longest_name_len = max(max([len(name) for name in names]), 6)
        print(" "*longest_metric_name_len, "|", end=" ")
        for name in names:
            print(name + " "*(longest_name_len - len(name)), "|", end=" ")
        print()
        for name, metric in metrics:
            print(" "*(longest_metric_name_len - len(name)) + name, "|", end=" ")
            for a, b in elements:
                value = repr(round(metric(a, b), min(longest_name_len - 2, 4)))
                print(value + " "*(longest_name_len - len(value)), "|", end=" ")
            print()

    return append, table

compare_append, compare_table = compare_deferred([("TVD", total_variation_distance),
                                                  ("Wasserstein", wasserstein_dst_voxel),
                                                  ("GMSD", gradient_magnitude_similarity_deviation),
                                                  ("Q-Score", compute_q_score)])
compare_append("Gaussian (A)", [gaussian, blob])

# %%
gaussian_copy = gaussian.copy()
blob_copy = blob.copy()

threshold = 0.005 # taken from Auto-Thresholding (https://github.com/DrDongSi/Auto-Thresholding)
gaussian_copy[gaussian < threshold] = map.min()
blob_copy[blob < threshold] = map.min()

compare_append("Gaussian (A) Thold", [gaussian_copy, blob_copy])

# %%
gaussian_copy = gaussian.copy()
blob_copy = blob.copy()

gaussian_copy[blob < threshold] = map.min()
blob_copy[blob < threshold] = map.min()

compare_append("Gaussian (A) Thold Alt", [gaussian_copy, blob_copy])

# %%
gaussian_alt = create_ligand(blob.shape, atoms, blob.min(), blob.max())

compare_append("Gaussian (B)", [gaussian_alt, blob])

# %%
gaussian_alt_copy = gaussian_alt.copy()
blob_copy = blob.copy()

gaussian_alt_copy[gaussian_alt < threshold] = map.min()
blob_copy[blob < threshold] = map.min()

compare_append("Gaussian (B) Thold", [gaussian_alt_copy, blob_copy])

# %%
gaussian_alt_copy = gaussian_alt.copy()
blob_copy = blob.copy()

gaussian_alt_copy[blob < threshold] = map.min()
blob_copy[blob < threshold] = map.min()

compare_append("Gaussian (B) Thold Alt", [gaussian_alt_copy, blob_copy])

# %%
compare_table()

# %%

