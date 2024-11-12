import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py 
from scipy.ndimage import binary_dilation #Q-score
from scipy.spatial.distance import cdist #Hausdorff Distance
from scipy.stats import wasserstein_distance #Wasserstein Distance
from scipy.ndimage import sobel #Gradient Magnitude Similarity Deviation
from scipy.spatial.distance import directed_hausdorff #Hausdorff Distance
from scipy.stats import wasserstein_distance_nd



#opcen3d and geomloss
def xor_similarity(voxel1, voxel2):
    """
    Calculate the XOR similarity between two voxel arrays.

    The XOR similarity is defined as 1 minus the ratio of the number of differing elements
    (computed using the logical XOR operation) to the total number of elements.

    Parameters:
    voxel1 (np.ndarray): The first voxel array.
    voxel2 (np.ndarray): The second voxel array.

    Returns:
    float: The XOR similarity between the two voxel arrays, ranging from 0 to 1.
    """
    xor_count = np.logical_xor(voxel1, voxel2).sum()
    total_voxels = voxel1.size
    return 1 - xor_count / total_voxels


#Doesnt work on pretty blobs due to size. cant allocate 160gb of memory.
def hausdorff_distance(voxel1, voxel2):
    """
    Calculate the Hausdorff distance between two 3D voxel grids.

    The Hausdorff distance is the maximum distance of a set to the nearest point in the other set.
    It measures how far the two voxel grids are from each other.

    Parameters:
    voxel1 (numpy.ndarray): A 3D numpy array representing the first voxel grid.
    voxel2 (numpy.ndarray): A 3D numpy array representing the second voxel grid.

    Returns:
    float: The Hausdorff distance between the two voxel grids.
    """
    # Get coordinates of occupied voxels in both grids
    coords1 = np.array(np.nonzero(voxel1)).T
    coords2 = np.array(np.nonzero(voxel2)).T

    # Compute distances from each point in voxel1 to the closest point in voxel2
    dists_1_to_2 = cdist(coords1, coords2, 'euclidean').min(axis=1)
    dists_2_to_1 = cdist(coords2, coords1, 'euclidean').min(axis=1)

    # Hausdorff distance is the max of these minimum distances
    hausdorff_dist = max(dists_1_to_2.max(), dists_2_to_1.max())
    return hausdorff_dist


def Hausdorff_dist(vol_a,vol_b):
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = 1000.0        
        for idx2 in range(len(vol_b)):
            dist= np.linalg.norm(vol_a[idx]-vol_b[idx2]) #euclidean distance
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)


def Hausdorff_normalized(voxel1, voxel2): 
    D = np.sqrt(voxel1.shape[0]**2 + voxel1.shape[1]**2 + voxel1.shape[2]**2)
    distance = Hausdorff_dist(voxel1, voxel2)
    return distance / D




#may work or may not
def hausdorff_distance_voxel(voxel1, voxel2):
    """
    Calculate the Hausdorff distance between two 3D voxel grids using scipy's directed Hausdorff.

    Parameters:
    voxel1 (numpy.ndarray): A 3D numpy array representing the first voxel grid.
    voxel2 (numpy.ndarray): A 3D numpy array representing the second voxel grid.

    Returns:
    float: The Hausdorff distance between the two voxel grids.
    """
    # Get coordinates of occupied voxels for non-binary values
    coords1 = np.argwhere(voxel1 > 0)
    coords2 = np.argwhere(voxel2 > 0)
    
    # Calculate directed Hausdorff distance from coords1 to coords2 and vice versa
    d1 = directed_hausdorff(coords1, coords2)[0]
    d2 = directed_hausdorff(coords2, coords1)[0]
    
    # Hausdorff distance is the maximum of the two directed distances
    hausdorff_dist = max(d1, d2)
    return hausdorff_dist





def total_variation_distance(voxel1, voxel2):
    """
    Calculate the Total Variation Distance (TVD) between two voxel arrays.

    The Total Variation Distance is a measure of the difference between two 
    probability distributions. It is used to measure the difference between two voxel arrays.

    Parameters:
    voxel1 (np.ndarray): The first voxel array (3D).
    voxel2 (np.ndarray): The second voxel array (3D).

    Returns:
    float: The Total Variation Distance between the two voxel arrays.
    """
    # Compute absolute difference and normalize
    abs_difference = np.abs(voxel1.astype(float) - voxel2.astype(float))
    tvd = np.sum(abs_difference) / voxel1.size
    return tvd
#zeby nie pracowac na 0.000x mozna dzielic przez ilosc voxeli niezerowych instead. 


def wasserstein_dst_voxel(voxel1, voxel2):
    """
    Compute the Wasserstein distance between two voxel grids.
    The function flattens the 3D voxel grids into 1D arrays and then computes the 
    Wasserstein distance between these arrays.
    Parameters:
    voxel1 (numpy.ndarray): The first voxel grid.
    voxel2 (numpy.ndarray): The second voxel grid.
    Returns:
    float: The Wasserstein distance between the two voxel grids.
    """
    # Flatten the voxel grids to 1D arrays
    voxel1_flat = voxel1.ravel().astype(float)
    voxel2_flat = voxel2.ravel().astype(float)
    
    # Compute Wasserstein distance
    wasserstein_dist = wasserstein_distance(voxel1_flat, voxel2_flat)
    return wasserstein_dist


def wasser(voxel1, voxel2):
    return wasserstein_distance_nd(voxel1, voxel2)

def gradient_magnitude_similarity_deviation(voxel1, voxel2):
    """
    Compute the Gradient Magnitude Similarity Deviation (GMSD) between two voxel grids.
    Parameters:
    voxel1 (ndarray): The first 3D voxel grid.
    voxel2 (ndarray): The second 3D voxel grid.
    Returns:
    float: The GMSD value, which is a measure of the similarity between the gradient magnitudes of the two voxel grids.
           A value closer to 1 indicates higher similarity, while a value closer to 0 indicates lower similarity.
    Notes:
    - The function computes the gradient magnitude of each voxel grid using the Sobel operator along each axis.
    - The absolute difference between the gradient magnitudes of the two voxel grids is then computed.
    - The GMSD is calculated as 1 minus the mean of the normalized absolute differences.
    - A small constant (1e-6) is added to the denominator to avoid division by zero.
    """
    # Compute the gradient magnitude of each voxel grid
    gradient_magnitude1 = np.sqrt(sum(sobel(voxel1, axis=i)**2 for i in range(3)))
    gradient_magnitude2 = np.sqrt(sum(sobel(voxel2, axis=i)**2 for i in range(3)))
    
    # Compute the absolute difference of the gradient magnitudes
    gmsd = 1- np.mean(np.abs((gradient_magnitude1 - gradient_magnitude2) / (gradient_magnitude1 + gradient_magnitude2 + 1e-6)))
    return gmsd

# https://github.com/jamaliki/qscore/blob/main/qscore/q_score.py
#scipy hausdorff doesnt work, its for 2D. found something on stack overflow with bbox and it also didnt work due to dimensions(?)


def qscore_similarity(voxel_grid1, voxel_grid2, threshold=0.1):
    """
    Compute the Q-score similarity measure between two voxel grids.
    
    Parameters:
    - voxel_grid1, voxel_grid2: 3D numpy arrays representing the voxel grids.
    - threshold: Minimum voxel intensity considered in the comparison (to ignore noise or low-density regions).
    
    Returns:
    - Q-score similarity between the two voxel grids.
    """
    # Ensure both grids are the same shape
    if voxel_grid1.shape != voxel_grid2.shape:
        raise ValueError("Voxel grids must have the same shape for comparison.")
    
    # Flatten the voxel grids for element-wise operations
    flat_grid1 = voxel_grid1.flatten()
    flat_grid2 = voxel_grid2.flatten()
    
    # Filter out low-intensity voxels based on the threshold
    mask = (flat_grid1 > threshold) & (flat_grid2 > threshold)
    grid1_filtered = flat_grid1[mask]
    grid2_filtered = flat_grid2[mask]
    
    # Calculate Q-score as the Pearson correlation coefficient
    if len(grid1_filtered) > 0:
        q_score = np.corrcoef(grid1_filtered, grid2_filtered)[0, 1]
    else:
        # If no high-density overlap, Q-score is zero
        q_score = 0.0
    
    return 1 - q_score



def compute_q_score(voxel1: np.array, voxel2: np.array) -> float:
    assert voxel1.shape == voxel2.shape, "diff in shape"
    voxel1 = voxel1.copy()
    voxel2 = voxel2.copy()

    v1_mean = np.mean(voxel1)
    v2_mean = np.mean(voxel2)

    u_normalized = voxel1 - v1_mean
    v_normalized = voxel2 - v2_mean

    numerator = np.sum(u_normalized * v_normalized)
    denominator = np.linalg.norm(u_normalized) * np.linalg.norm(v_normalized)

    if denominator == 0:
        return 0

    q_score_value = numerator / denominator
    return abs(1 - q_score_value)