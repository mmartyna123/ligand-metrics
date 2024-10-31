import numpy as np
from utils import *
from skimage import measure
import napari
import matplotlib.pyplot as plt

def load_ligands(cif_file):
    """
    Input:
        cif_file - path to a cif file containing ligands
    Return:
        ligands - list of ligands
    """
    return extract_ligand_coords(cif_file)[0]

def load_blob(blob_file):
    """
    Input:
        blob_file - path to a npz file containing a blob
    Return:
        blob - voxel grind of the blob
    """
    return np.load(blob_file)["arr_0"]

def blobify(ligand, resolution=1, padding=0):
    """
    Input:
        ligand - list of tuples of atom names and atom positions in Arnstroms
        resolution - how much to scale up the ligand (default: 1)
        padding - how much space to leave around the ligand in Arnstroms (default: 0)
    Return:
        blob - voxel grid with values from 0 to 1 masking the shape of the ligand
    """
    offset, size = get_offset_and_size(ligand)

    blob = np.zeros(shape=np.ceil((size + 2*padding) * resolution).astype(int))
    for atom_name, position in ligand:
        center = (position - offset + padding) * resolution
        radius = get_atomic_radius(atom_name) * resolution
        blob = np.logical_or(blob, sphere(blob.shape, radius, center))

    return blob

def blobify_like(ligand, other_blob):
    """
    Input:
        ligand - list of tuples of atom names and atom positions in Arnstroms
        other_blob - voxel grind containing a shape
    Return:
        blob - numpy array with values from 0 to 1 masking the shape of the ligand
    """
    offset, size = get_offset_and_size(ligand)

    x, y, z = cut_blob.nonzero()
    bb_min = np.array([x.min(), y.min(), z.min()])
    bb_max = np.array([x.max(), y.max(), z.max()])
    scale = (bb_max - bb_min) / size

    blob = np.zeros_like(other_blob)
    for atom_name, position in ligand:
        center = (position - offset) * scale + bb_min
        radius = get_atomic_radius(atom_name) * np.mean(scale)
        blob = np.logical_or(blob, sphere(blob.shape, radius, center))

    return blob

def plot_blob(blob, fig=None, size=1, idx=1, share_ax=None):
    """
    Input:
        blob - voxel grid containing a shape
    """
    fig = plt.figure() if fig is None else fig

    ax = fig.add_subplot(1, size, idx, projection='3d', sharex=share_ax, sharey=share_ax, sharez=share_ax)

    verts, faces, normals, values = measure.marching_cubes(blob, 0)
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
        antialiased=False, linewidth=0.0)

    return fig, ax

def plot_blob_comparison(blob, other_blob):
    """
    Input:
        blob - voxel grid containing a shape
        other_blob - voxel grind containing a shape to compare with
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    axes = []

    axes.append(plot_blob(blob, fig, 2, 1)[1])
    axes.append(plot_blob(other_blob, fig, 2, 2, axes[0])[1])

    return fig, axes

def napari_blob_comparison(blob, other_blob):
    """
    Input:
        blob - voxel grid containing a shape
        other_blob - voxel grind containing a shape to compare with
    """
    viewer = napari.Viewer()
    viewer.add_image(blob, name="blob", colormap="bop blue")
    viewer.add_image(other_blob, name="other blob", opacity=0.5, colormap="bop orange")
    return viewer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ligand", help="ligand to blobify and display")
    parser.add_argument("-c", "--compare", action="store_true", help="compare with cut blob")
    args = parser.parse_args()
    pdb_id = args.ligand.partition("_")[0]

    ligands = load_ligands(f"{pdb_id}.cif")
    ligand = ligands[args.ligand]
    if args.compare:
        cut_blob = load_blob(f"{args.ligand}.npz")
        perfect_blob = blobify_like(ligand, cut_blob)
        plot_blob_comparison(perfect_blob, cut_blob)
    else:
        perfect_blob = blobify(ligand, resolution=5, padding=1)
        plot_blob(perfect_blob)
    plt.show()

