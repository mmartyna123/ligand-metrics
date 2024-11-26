# %%
from blobify import *
from tqdm import tqdm

map, scale, origin = load_map("6HCY.mrc.gz")
blob = load_blob("ours_6HCY_C_502_HEM.npz")
threshold = 0.005 # taken from Auto-Thresholding (https://github.com/DrDongSi/Auto-Thresholding)
ligands = load_ligands("6HCY.cif")
ligand = ligands["6HCY_C_502_HEM"]
offset, size = get_offset_and_size(ligand)
atoms = ligand_to_atoms(ligand, offset, scale, origin)
len(atoms)

# %%
def sphere_points(position, radius, num_points):
    """
    Generates points evenly distributed on the surface of a sphere.

    Parameters:
    position (tuple): A tuple of three floats representing the (x, y, z) coordinates of the sphere's center.
    radius (float): The radius of the sphere.
    num_points (int): The number of points to generate on the sphere's surface.

    Returns:
    list: A list of tuples representing the coordinates of the points on the sphere.
    """
    x_center, y_center, z_center = position
    points = []

    for i in range(num_points):
        phi = np.arccos(1 - 2 * (i + 0.5) / num_points)  # polar angle
        theta = np.pi * (1 + 5**0.5) * i  # azimuthal angle

        x = x_center + radius * np.sin(phi) * np.cos(theta)
        y = y_center + radius * np.sin(phi) * np.sin(theta)
        z = z_center + radius * np.cos(phi)

        points.append((x, y, z))

    return points

def min_distance(point, atoms):
    return min([np.linalg.norm(point - center) - radius.mean() for center, radius in atoms])

def get_radial_points(atom, other_atoms, radius, min_num_points):
    result = None
    for try_n in range(0, 99): # give up after 100 tries and keep possibly fewer than N points
        result = []

        points = sphere_points(atom[0], radius, min_num_points + try_n)
        for point in points:
            if min_distance(point, [atom]) < min_distance(point, other_atoms):
                result.append(point)

        if len(result) >= min_num_points:
            break

    return result

get_radial_points(atoms[0], atoms[1:], 1, 9)

# %%
def get_reference_gaussian_params(map):
    # determine max and min value in map M
    map_values = map.reshape(-1)
    max_m = max(map_values)
    min_m = min(map_values)
    # determine value 10 standard deviations above mean (capped at max_m)
    high_v = min(np.mean(map_values) + np.std(map_values)*10, max_m)
    # determine value 1 standard deviations below mean (capped at min_m)
    low_v = max(np.mean(map_values) - np.std(map_values)*1, min_m)
    # determine reference gaussian height, A, and offset, B
    A = high_v - low_v
    B = low_v
    return A, B

def map_value_at_point(map, point):
    """
    Interpolates the value at a given point based on nearby grid voxels.

    Parameters:
    map (numpy.ndarray): A 3D numpy array representing the voxel grid.
    point (tuple): A tuple of three floats representing the (x, y, z) coordinates of the point.

    Returns:
    float: The interpolated value at the specified point.
    """

    x_dim, y_dim, z_dim = map.shape
    x, y, z = point

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    z0 = int(np.floor(z))
    z1 = z0 + 1

    if (x0 < 0 or x1 >= x_dim or
        y0 < 0 or y1 >= y_dim or
        z0 < 0 or z1 >= z_dim):
        raise ValueError("Point is out of bounds of the map.")

    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Perform trilinear interpolation
    c000 = map[x0, y0, z0]
    c100 = map[x1, y0, z0]
    c010 = map[x0, y1, z0]
    c110 = map[x1, y1, z0]

    c001 = map[x0, y0, z1]
    c101 = map[x1, y0, z1]
    c011 = map[x0, y1, z1]
    c111 = map[x1, y1, z1]

    # Interpolate along x
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # Interpolate along y
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z
    interpolated_value = c0 * (1 - zd) + c1 * zd

    return interpolated_value

def correlation_about_mean(a, b):
    """
    Calculates the Pearson correlation coefficient between two lists of values using NumPy.

    Parameters:
    a (np.array): First list of values.
    b (np.array): Second list of values (must be of equal length to a).

    Returns:
    float: The Pearson correlation coefficient between a and b.
    """

    if a.shape[0] != b.shape[0]:
        raise ValueError("Lists a and b must have the same length.")

    # Calculate the Pearson correlation coefficient
    correlation_matrix = np.corrcoef(a, b)

    # Return the correlation coefficient (off-diagonal element)
    return correlation_matrix[0, 1]

    # u_normalized = a - np.mean(a)
    # v_normalized = b - np.mean(b)
    #
    # numerator = np.sum(u_normalized * v_normalized)
    # denominator = np.linalg.norm(u_normalized) * np.linalg.norm(v_normalized)
    #
    # if denominator == 0:
    #     return 0
    #
    # q_score_value = numerator / denominator
    # return q_score_value

def calculate_q_score_for_atom(atom, other_atoms, map, A, B, sigma, num_pts):
    map_values = []
    map_value_at_R0 = map_value_at_point(map, atom[0])
    map_values.extend([map_value_at_R0] * num_pts)

    reference_gaussian_values = []
    reference_gaussian_value_at_R0 = A + B
    reference_gaussian_values.extend([reference_gaussian_value_at_R0] * num_pts)

    for radius in np.arange(0.1, 2.0, 0.1):
        radial_points = get_radial_points(atom, other_atoms, radius, num_pts)
        map_values_at_points = [map_value_at_point(map, point) for point in radial_points]
        map_values.extend(map_values_at_points)

        reference_gaussian_value_at_R = A * np.e**(-(1/2)*(radius/sigma)**2) + B
        reference_gaussian_values_at_R = [reference_gaussian_value_at_R] * len(radial_points)
        reference_gaussian_values.extend(reference_gaussian_values_at_R)

    atomQ = correlation_about_mean(np.array(map_values),
                                   np.array(reference_gaussian_values))
    return atomQ

def calculate_q_score(map, atoms, A, B, sigma=0.6, num_pts=8):
    sum = 0
    for atom in tqdm(atoms):
        others = [other_atom for other_atom in atoms if other_atom is not atom]
        atomQ = calculate_q_score_for_atom(atom, others, map, A, B, sigma, num_pts)
        sum += atomQ
    return sum / len(atoms)

A, B = get_reference_gaussian_params(map)
atoms_in_map_space = [(center + offset * scale, radius) for center, radius in atoms]
print("q-score      ", calculate_q_score(map, atoms_in_map_space, A, B, sigma=1))
print("q-score thold", calculate_q_score(binarize(map, threshold) * map, atoms_in_map_space, A, B, sigma=1))

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

gaussian = create_ligand(blob.shape, atoms, blob.min(), blob.max())
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
compare_append("Gaussian", [gaussian, blob])

# %%
gaussian_copy = gaussian.copy()
blob_copy = blob.copy()

gaussian_copy[gaussian < threshold] = map.min()
blob_copy[blob < threshold] = map.min()

compare_append("Gaussian Thold", [gaussian_copy, blob_copy])

# %%
gaussian_copy = gaussian.copy()
blob_copy = blob.copy()

gaussian_copy[blob < threshold] = map.min()
blob_copy[blob < threshold] = map.min()

compare_append("Gaussian Thold Alt", [gaussian_copy, blob_copy])

# %%
compare_table()

# %%

