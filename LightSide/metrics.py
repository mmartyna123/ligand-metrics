import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py 
from scipy.ndimage import binary_dilation #Q-score
from scipy.spatial.distance import cdist #Hausdorff Distance
from scipy.stats import wasserstein_distance #Wasserstein Distance
from scipy.ndimage import sobel #Gradient Magnitude Similarity Deviation
from scipy.stats import wasserstein_distance_nd
from scipy.sparse import lil_matrix
import ot
from scipy.sparse import csr_matrix


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
    return  xor_count / total_voxels


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
    return 1 - tvd
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
    return 1 - wasserstein_dist


# def wasser(voxel1, voxel2):
#     return wasserstein_distance_nd(voxel1, voxel2)

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
    gmsd = np.mean(np.abs((gradient_magnitude1 - gradient_magnitude2) / (gradient_magnitude1 + gradient_magnitude2 + 1e-6)))
    return 1 - gmsd

# https://github.com/jamaliki/qscore/blob/main/qscore/q_score.py
#scipy hausdorff doesnt work, its for 2D. found something on stack overflow with bbox and it also didnt work due to dimensions(?)





def compute_q_score(voxel1: np.array, voxel2: np.array) -> float:
    assert voxel1.shape == voxel2.shape, "diff in shape"
    voxel1 = voxel1.copy()
    voxel2 = voxel2.copy()
    
    # if np.min(voxel1) == np.max(voxel1) or np.min(voxel2) == np.max(voxel2):
    #     return "Deal with it"
    
    #mask = (voxel1 != 0) | (voxel2 != 0)
    
    #v1_mean = np.mean(voxel1[mask])
    #v2_mean = np.mean(voxel2[mask])

    #u_normalized = (voxel1[mask] - np.min(voxel1[mask])) / (np.max(voxel1[mask]) - np.min(voxel1[mask]))
    #v_normalized = (voxel2[mask] - np.min(voxel2[mask])) / (np.max(voxel2[mask]) - np.min(voxel2[mask]))
    

    # u_normalized = (voxel1 - np.min(voxel1)) / (np.max(voxel1) - np.min(voxel1))
    # v_normalized = (voxel2 - np.min(voxel2)) / (np.max(voxel2) - np.min(voxel2))
    u_normalized = voxel1 - np.mean(voxel1)
    v_normalized = voxel2 - np.mean(voxel2)

    numerator = np.sum(u_normalized * v_normalized)
    denominator = np.linalg.norm(u_normalized) * np.linalg.norm(v_normalized)

    if denominator == 0:
        return 0

    q_score_value = numerator / denominator
    # print(numerator, denominator)
    return q_score_value


####

def wasserstein_distance_3d(grid1, grid2):
    # Get the indices of non-zero values
    indices1 = np.argwhere(grid1 != 0)
    indices2 = np.argwhere(grid2 != 0)

    # Get the values at these indices
    values1 = grid1[grid1 != 0]
    values2 = grid2[grid2 != 0]
    
    if values1.size == 0 or values2.size == 0:
        return 1.0 if values1.size == values2.size else 0.0

    # Normalize the values to sum to 1
    a = values1 / np.sum(values1)
    b = values2 / np.sum(values2)

    # Compute pairwise distances between the selected indices
    cost_matrix = cdist(indices1, indices2, metric='euclidean')

    # Solve the optimal transport problem
    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=0.1)

    wasserstein_distance = np.sum(transport_plan * cost_matrix)
    
    max_distance = np.sqrt(np.sum(np.maximum(grid1.shape, grid2.shape) ** 2))
    
    similarity = 1 - (wasserstein_distance / max_distance)

    
    return  similarity



def wasserstein_distance_3d_optimized(grid1, grid2, reg=0.1, threshold=0.01, downsample_factor=2):
    """
    Compute the Wasserstein similarity (0 to 1) between two 3D voxel grids using an optimized approach.
    
    Parameters:
        grid1 (np.ndarray): First 3D voxel grid.
        grid2 (np.ndarray): Second 3D voxel grid.
        reg (float): Regularization parameter for the Sinkhorn algorithm.
        threshold (float): Threshold to ignore low-intensity voxels.
        downsample_factor (int): Factor by which to downsample the grids for efficiency.
    
    Returns:
        float: Normalized Wasserstein similarity (1 = perfect match, 0 = no match).
    """
    # Downsample the grids
    grid1_down = grid1[::downsample_factor, ::downsample_factor, ::downsample_factor]
    grid2_down = grid2[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Extract significant voxels (values > threshold)
    indices1 = np.argwhere(grid1_down > threshold)
    indices2 = np.argwhere(grid2_down > threshold)
    values1 = grid1_down[grid1_down > threshold]
    values2 = grid2_down[grid2_down > threshold]

    # Handle edge cases where one or both grids are empty
    if values1.size == 0 or values2.size == 0:
        return 1.0 if values1.size == values2.size else 0.0

    # Normalize the voxel values to sum to 1
    a = values1 / values1.sum()
    b = values2 / values2.sum()

    # Compute the cost matrix (pairwise Euclidean distances)
    cost_matrix = cdist(indices1, indices2, metric='euclidean')

    # Solve the optimal transport problem using Sinkhorn algorithm
    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=reg)
    wasserstein_distance = np.sum(transport_plan * cost_matrix)

    # Compute the maximum possible distance for normalization
    max_distance = np.sqrt(np.sum(np.maximum(grid1.shape, grid2.shape) ** 2))

    # Normalize Wasserstein distance to obtain similarity
    similarity = 1 - (wasserstein_distance / max_distance)
    # print(wasserstein_distance, max_distance)
    return similarity



def wasserstein_distance_3d_optimized_normalized(grid1, grid2, reg=0.1, threshold=0.01, downsample_factor=2):
    """
    Compute the Wasserstein similarity (0 to 1) between two 3D voxel grids using an optimized approach.
    
    Parameters:
        grid1 (np.ndarray): First 3D voxel grid.
        grid2 (np.ndarray): Second 3D voxel grid.
        reg (float): Regularization parameter for the Sinkhorn algorithm.
        threshold (float): Threshold to ignore low-intensity voxels.
        downsample_factor (int): Factor by which to downsample the grids for efficiency.
    
    Returns:
        float: Normalized Wasserstein similarity (1 = perfect match, 0 = no match).
    """
    #downsample the grids
    grid1_down = grid1[::downsample_factor, ::downsample_factor, ::downsample_factor]
    grid2_down = grid2[::downsample_factor, ::downsample_factor, ::downsample_factor]

    #extracting only the  significant voxels where values > threshold
    indices1 = np.argwhere(grid1_down > threshold)
    indices2 = np.argwhere(grid2_down > threshold)
    values1 = grid1_down[grid1_down > threshold]
    values2 = grid2_down[grid2_down > threshold]

    #handling edge cases where one or both grids are empty
    if values1.size == 0 or values2.size == 0:
        return 1.0 if values1.size == values2.size else 0.0

    #normalizing the voxel values to sum to 1
    a = values1 / values1.sum()
    b = values2 / values2.sum()

    # computing the cost matrix (pairwise Euclidean distances)
    cost_matrix = cdist(indices1, indices2, metric='euclidean')

    #transport problem using Sinkhorn algorithm
    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=reg)
    wasserstein_distance = np.sum(transport_plan * cost_matrix)

    # computing max distance between the non-zero regions for normalization (between the two grids)
    combined_indices = np.vstack([indices1, indices2]) # Combine indices from both grids
    max_distance = np.sqrt(np.sum((combined_indices.max(axis=0) - combined_indices.min(axis=0)) ** 2))

    #fallback for very sparse grids (to avoid zero max_distance)
    max_distance = max(max_distance, 1e-8)

    # NORMALIZATION -- IS THERE BETTER APPROACH ??
    similarity = 1 - (wasserstein_distance / max_distance)
    # print(wasserstein_distance, max_distance)
    return similarity



def wasserstein_distance_3d_average_distance(grid1, grid2, reg=0.1, threshold=0.01, downsample_factor=2):
    grid1_down = grid1[::downsample_factor, ::downsample_factor, ::downsample_factor]
    grid2_down = grid2[::downsample_factor, ::downsample_factor, ::downsample_factor]

    indices1 = np.argwhere(grid1_down > threshold)
    indices2 = np.argwhere(grid2_down > threshold)
    values1 = grid1_down[grid1_down > threshold]
    values2 = grid2_down[grid2_down > threshold]

    if values1.size == 0 or values2.size == 0:
        return 1.0 if values1.size == values2.size else 0.0

    a = values1 / values1.sum()
    b = values2 / values2.sum()

    cost_matrix = cdist(indices1, indices2, metric='euclidean')
    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg=reg)
    wasserstein_distance = np.sum(transport_plan * cost_matrix)

    # Normalize by average pairwise distances of voxels
    avg_distance1 = np.mean(cdist(indices1, indices1, metric='euclidean'))
    avg_distance2 = np.mean(cdist(indices2, indices2, metric='euclidean'))
    avg_distance = max(avg_distance1, avg_distance2)

    max_distance = max(avg_distance, 1e-8)  # Avoid division by zero
    similarity = 1 - (wasserstein_distance / max_distance)
    return similarity
