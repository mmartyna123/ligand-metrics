# Blobifying ligands
## Prerequisites
Install package and OpenCL driver for your system
```sh
pip install -e .
```

## Usage
### Batch
This tool is designed to mass process density maps.

First, create `data.yaml` to store map and structure data.
```yaml
data:
  - emd: 0199
    thr: 0.005
    pdb: 6hcy
    res: 3.1
```
Then, run the `batch.py` script.
```sh
python batch.py --verbose --jobs 4 data.yaml
```

### Cutter
This tool is designed to cut out the density around studied ligands in a structure.

First, download the structure and the density map.
```sh
wget https://files.rcsb.org/download/6HCY.cif
wget https://files.rcsb.org/pub/emdb/structures/EMD-0199/map/emd_0199.map.gz
```
Then, run the `cutter.py` script.
```sh
python cutter.py --verbose 6HCY.cif emd_0199.map.gz
```

### Blobify
This tool is designed to plot the density of a studied ligand.

First, download the structure.
```sh
wget https://files.rcsb.org/download/6HCY.cif
```
Then, run the `blobify.py` script.
```sh
python blobify.py 6HCY_C_502_HEM
```
