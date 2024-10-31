# Blobifying ligands
## Prerequisites
Install package
```sh
pip install -e .
```

Get [the CIF file](https://www.rcsb.org/structure/6G2J)
```sh
wget https://files.rcsb.org/download/6G2J.cif
```

Get [the cut blob (6G2J_M_501_3PE.npz)](https://drive.google.com/file/d/1SaPA1f6Z4dO19KnEVuZTONCEQJGgi5S1/view?usp=sharing)

## Usage
Import it and use the provided functions
```python
from blobify import *

ligands = load_ligands("6G2J.cif")
ligand = ligands["6G2J_M_501_3PE"]
cut_blob = load_blob("6G2J_M_501_3PE.npz")
perfect_blob = blobify_like(ligand, cut_blob)
plot_blob_comparison(perfect_blob, cut_blob)
plt.show()
```

## Run
You can run it with python
```sh
python blobify.py
```
and it should display two blobs side by side

