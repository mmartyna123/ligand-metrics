import numpy as np
from utils import *
from skimage import measure
import napari
import matplotlib.pyplot as plt

# %%
ligands, resolution, num_particles = extract_ligand_coords("6G2J.cif")
ligands.keys(), resolution, num_particles

# %%
# (pdb id)_(chain id)_(residue id[1])_(residue id[0][2:])
ligand = ligands["6G2J_M_501_3PE"]
ligand

# %%
for element in get_unique_elements(ligand):
    print(element.name, f"{element.atomic_radius}pm")

# %%
offset, size = get_offset_and_size(ligand)
offset, size

# %%
cut_blob = np.load("6G2J_M_501_3PE.npz")["arr_0"]
cut_blob.shape

# %%
x, y, z = cut_blob.nonzero()
bb_min, bb_max = np.array([x.min(), y.min(), z.min()]), np.array([x.max(), y.max(), z.max()])
bb_min, bb_max

# %%
scale = (bb_max - bb_min) / size
scale

# %%
perfect_blob = np.zeros_like(cut_blob)
for atom_name, coord in ligand:
    center = (coord - offset) * scale + bb_min
    # TODO: figure out a how to scale the atomic_radius
    atomic_radius = cif_atom_to_mendeleev_element(atom_name).atomic_radius * 0.075 # 0.1
    perfect_blob = np.logical_or(perfect_blob, sphere(perfect_blob.shape, atomic_radius, center))

# %%
fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1, projection='3d')

verts, faces, normals, values = measure.marching_cubes(perfect_blob, 0)
ax.plot_trisurf(
    verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
    antialiased=False, linewidth=0.0)

ax = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax, sharey=ax, sharez=ax)

verts, faces, normals, values = measure.marching_cubes(cut_blob, 0)
ax.plot_trisurf(
    verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
    antialiased=False, linewidth=0.0)

plt.show()

# %%
viewer = napari.Viewer()
viewer.add_image(cut_blob, name="cut blob", colormap="bop blue")
viewer.add_image(perfect_blob, name="perfect blob", opacity=0.5, colormap="bop orange")

# %%

