from blobify import *

# %%
ligands = load_ligands("6HCY.cif")
len(ligands)

# %%
ligand = ligands["6HCY_C_502_HEM"]
fig, ax = plot_blob(blobify(ligand, 5, 1))
set_axes_unit([ax], 5)
plt.show()

# %%
import mrcfile

mrc = mrcfile.open("6HCY.mrc.gz")
mrc

# %%
mrc.print_header()

# %%
mrc.is_volume()

# %%
cell_a = mrc.header.cella[["x","y","z"]].view(("f4", 3))
cell_a

# %%
cell_b = mrc.header.cellb[["alpha","beta","gamma"]].view(("f4", 3))
cell_b

# %%
origin = np.array([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart]).astype("f4")
origin

# %%
mrc.data.shape

# %%
mrc.voxel_size

# %%
order = (3 - mrc.header.maps, 3 - mrc.header.mapr, 3 - mrc.header.mapc)
map = np.asarray(mrc.data, dtype="f4")
map = np.moveaxis(a=map, source=order, destination=(2, 1, 0))
map.shape

# %%
scale = map.shape / cell_a

offset, size = get_offset_and_size(ligand)
size = np.ceil(size * scale).astype(int)

blob = np.zeros(shape=size)
for atom_name, position in ligand:
    center = (position - offset) * scale
    radius = get_atomic_radius(atom_name) * np.mean(scale)
    blob = np.logical_or(blob, sphere(blob.shape, radius, center))

offset = np.floor(offset * scale).astype(int)

fig, ax = plot_blob(blob)
set_axes_unit([ax], np.mean(scale))
plt.show()

# %%
cutout = map[offset[0]:offset[0]+size[0],offset[1]:offset[1]+size[1],offset[2]:offset[2]+size[2]]
cutout.shape

# %%
plt.hist(cutout.reshape(-1), bins=200)
plt.show()

# %%
cutout.mean()

# %%
mask = blobify_like(ligand, cutout)
fig, ax = plot_blob(mask)
set_axes_unit([ax], np.mean(scale))
plt.show()

# %%
napari_blob_comparison(mask, cutout)

# %%
fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True)

axs[0].hist(cutout.reshape(-1), bins=50)
axs[1].hist(cutout[mask.nonzero()].reshape(-1), bins=50)

plt.show()

# %%
from scipy import stats

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True)

x = np.arange(cutout.min(), cutout.max(), 0.001)
axs[0].plot(x, stats.gaussian_kde(cutout.reshape(-1))(x))
axs[1].plot(x, stats.gaussian_kde(cutout[mask.nonzero()].reshape(-1))(x))

plt.show()

# %%
cutout.mean(), cutout[mask.nonzero()].mean()

# %%
blob = cutout.copy()

threshold = 0.12 # taken visually from chimerax
blob[blob >= threshold] = 1
blob[blob <  threshold] = 0

fig, axes = plot_blob_comparison(mask, blob)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("cif", "6HCY.cif")

pdb_id = "6HCY"
model = structure[0]
all_atoms = {"not_studied": []}

for chain in model:
    chain_id = chain.get_id()

    for residue in chain:
            ligand_coords = []
            for atom in residue:
                ligand_coords.append((atom.element, atom.get_coord()))

            if is_studied_ligand(residue):
                ligand_name = f"{pdb_id}_{chain_id}_{residue.get_id()[1]}_{residue.get_id()[0][2:]}"
                all_atoms[ligand_name] = ligand_coords
            else:
                all_atoms["not_studied"] += ligand_coords

len(all_atoms), len(all_atoms["not_studied"])

# %%
from scipy.spatial import KDTree

ligands_pc = []
ligands_names = []
ligands_atoms = []
for ligand_name in all_atoms:
    for element_name, position in all_atoms[ligand_name]:
        center = position - origin
        radius = get_atomic_radius(element_name)
        ligands_pc.append((center - radius) * scale)
        ligands_pc.append((center + radius) * scale)
        ligands_atoms.append((center * scale, radius * scale))
        ligands_atoms.append((center * scale, radius * scale))
        ligands_names.append(ligand_name)
        ligands_names.append(ligand_name)

lookup = KDTree(ligands_pc)

# ideally the KDTree would implement query_rectangle([offset, offset + size])
center = offset + size / 2
radius = np.ceil((size @ size) ** 0.5 * 0.5)

ligand_atoms_idxs = [idx for idx, ligand_name in enumerate(ligands_names) if ligand_name == "6HCY_C_502_HEM"]
neighbour_atoms_idxs = lookup.query_ball_point(center, radius)
neighbour_atoms_idxs = list(filter(lambda idx: not idx in ligand_atoms_idxs, neighbour_atoms_idxs))
print({ligands_names[idx] for idx in neighbour_atoms_idxs})
len(neighbour_atoms_idxs) # TODO: this contains duplicates (bb_min, bb_max)

# %%
offset, size = get_offset_and_size(ligand)
size = np.ceil(size * scale).astype(int)

neighbor_atoms = [(ligands_atoms[idx][0] - offset*scale, ligands_atoms[idx][1]) for idx in neighbour_atoms_idxs]
bb_min = np.floor(np.array([center - radius for center, radius in neighbor_atoms]).min(axis=0)).astype(int)
bb_max = np.ceil(np.array([center + radius for center, radius in neighbor_atoms]).max(axis=0)).astype(int)

positive_mask = np.zeros(shape=size)
ligand_atoms = []
for atom_name, position in ligand:
    center = (position - offset) * scale
    radius = get_atomic_radius(atom_name) * scale
    ligand_atoms.append((center, radius))
    positive_mask = np.logical_or(positive_mask, sphere(positive_mask.shape, radius.mean(), center))

negative_mask = np.zeros(shape=bb_max - bb_min)
for center, radius in neighbor_atoms:
    negative_mask = np.logical_or(negative_mask, sphere(negative_mask.shape, np.mean(radius), center - bb_min))
offset = np.floor(offset * scale).astype(int)

negative_mask = negative_mask[-bb_min[0]:size[0]-bb_min[0], -bb_min[1]:size[1]-bb_min[1], -bb_min[2]:size[2]-bb_min[2]]

fig, axes = plot_blob_comparison(positive_mask, negative_mask)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
distance_positive_mask = np.empty(shape=size)
distance_negative_mask = np.empty(shape=size)

from tqdm import tqdm
for idx in tqdm(np.arange(stop=size[0]*size[1]*size[2])):
    x = idx % size[0]
    y = (idx // size[0]) % size[1]
    z = idx // (size[0] * size[1])
    position = np.array([x,y,z])
    distance_positive_mask[x,y,z] = min([np.linalg.norm(position - center) - radius.mean() for center, radius in ligand_atoms]) # positive atoms
    distance_negative_mask[x,y,z] = min([np.linalg.norm(position - center) - radius.mean() for center, radius in neighbor_atoms]) # negative atoms

# %%
blob = cutout.copy()

blob[distance_positive_mask > distance_negative_mask] = blob.min()

threshold = 0.1
blob[blob >= threshold] = 1
blob[blob <  threshold] = 0

fig, axes = plot_blob_comparison(positive_mask, blob)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
blob = cutout.copy()

blob[distance_positive_mask > distance_negative_mask] = blob.min()

np.savez("ours_6HCY_C_502_HEM.npz", blob)

# %%
blob = load_blob("ours_6HCY_C_502_HEM.npz")

fig, axex = plot_blob_comparison(mask, blob)
set_axes_unit(axes, scale)
plt.show()

# %%

