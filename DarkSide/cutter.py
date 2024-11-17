from blobify import *

# %%
map, scale, origin = load_map("6HCY.mrc.gz")
map.shape, scale, origin

# %%
ligands = load_ligands("6HCY.cif")
ligand = ligands["6HCY_C_502_HEM"]
offset, size = get_offset_and_size(ligand)
cutout = cut_out_map(map, scale, offset, size)
cutout.shape

# %%
blob = cutout.copy()

threshold = 0.12 # taken visually from chimerax
blob[blob >= threshold] = 1
blob[blob <  threshold] = 0

perfect = blobify_like(ligand, blob)
fig, axes = plot_blob_comparison(perfect, blob)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
lookup = get_lookup(ligands, origin, scale)
neighbor_atoms = lookup(offset, size, "6HCY_C_502_HEM")
len(neighbor_atoms)

# %%
ligand_atoms = ligand_to_atoms(ligand, offset, scale, origin)
len(ligand_atoms)

# %%
positive_mask = get_mask(ligand_atoms, cutout.shape)
negative_mask = get_mask(neighbor_atoms, cutout.shape)

fig, axes = plot_blob_comparison(positive_mask, negative_mask)
set_axes_unit(axes, np.mean(scale))
plt.show()

# %%
cl_ctx, calc_min_distance = init_cl()
distance_positive_mask = calc_min_distance(ligand_atoms, cutout.shape)
distance_negative_mask = calc_min_distance(neighbor_atoms, cutout.shape)
distance_positive_mask.mean(), distance_negative_mask.mean()

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

fig, axex = plot_blob_comparison(positive_mask, blob)
set_axes_unit(axes, scale)
plt.show()

# %%

