# Blobifying ligands
## Prerequisites
Install package
```sh
pip install -e .
```

Get [the CIF file](https://www.rcsb.org/structure/6HCY), [the cut blobs](https://zenodo.org/records/10908325) and extract `6HCY_C_502_HEM.npz`
```sh
wget https://files.rcsb.org/download/6HCY.cif
wget https://zenodo.org/records/10908325/files/cryoem_blobs.zip
unzip -p cryoem_blobs.zip cryoem_blobs/6HCY_C_502_HEM.npz > 6HCY_C_502_HEM.npz
```

## Usage
Import and use the provided functions
```python
from blobify import *

ligands = load_ligands("6HCY.cif")
ligand = ligands["6HCY_C_502_HEM"]
cut_blob = load_blob("6HCY_C_502_HEM.npz")
perfect_blob = blobify_like(ligand, cut_blob)
plot_blob_comparison(perfect_blob, cut_blob)
plt.show()
```

## Run
You can run it with python
```sh
python blobify.py 6HCY_C_502_HEM
```
to blobify the given ligand and display it

Or you can pass it the `--compare` flag to side by side
compare it with a cut ligand
```sh
python blobify.py --compare 6HCY_C_502_HEM
```

