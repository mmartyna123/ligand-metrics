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
map = np.moveaxis(a=map, source=(0, 1, 2), destination=order)
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

fig, ax = plot_blob(blob)
set_axes_unit([ax], np.mean(scale))
plt.show()

# %%
offset = np.floor(offset).astype(int)

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
from scipy import stats

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True)

x = np.arange(cutout.min(), cutout.max(), 0.001)
axs[0].plot(x, stats.gaussian_kde(cutout.reshape(-1))(x))
axs[1].plot(x, stats.gaussian_kde(cutout[mask.nonzero()].reshape(-1))(x))

plt.show()

# %%
blob = cutout.copy()

threshold = 0.045
blob[blob >= threshold] = 1
blob[blob <  threshold] = 0

fig, axes = plot_blob_comparison(mask, blob)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
blob = cutout.copy()

blob = blob.clip(cutout[mask.nonzero()].min(), cutout[mask.nonzero()].max())

fig, axes = plot_blob_comparison(mask, blob)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
cutout.mean(), cutout[mask.nonzero()].mean()

# %%
mean_blob = cutout.copy()
mean_blob[mean_blob >= cutout.mean()] = 1
mean_blob[mean_blob <  cutout.mean()] = 0

fig, axes = plot_blob_comparison(mask, blob)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%

