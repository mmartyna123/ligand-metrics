import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
import mendeleev
from functools import lru_cache

# ==== Handling ligands ==== #
@lru_cache
def get_atomic_radius(atom_name):
    "atomic radius in Angstrom"
    atom_name = atom_name[0].upper() + atom_name[1:].lower()
    return pm_to_angstrom(mendeleev.element(atom_name).atomic_radius)

def get_unique_elements(ligand):
    return list(set(map(lambda atom: atom[0], ligand)))

def pm_to_angstrom(pm):
    return pm * 0.01

def get_offset_and_size(ligand):
    bb_min, bb_max = np.ones(shape=(3,)) * float("inf"), np.ones(shape=(3,)) * float("-inf")
    for name, coord in ligand:
        atomic_radius = get_atomic_radius(name)
        bb_min = np.minimum(bb_min, coord - atomic_radius)
        bb_max = np.maximum(bb_max, coord + atomic_radius)
    return bb_min, bb_max - bb_min

# ==== Reading CIF files ==== #
# Borrowed from https://github.com/dabrze/cryo-em-ligand-cutter/blob/main/utils/cryoem_utils.py#L73-L159

def extract_ligand_coords(cif_file):
    """
    Extracts the coordinates of ligands and nearby atoms from a CIF file.

    Args:
        cif_file (str): The path to the CIF file.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of ligand names and their corresponding coordinates.
            - float: The resolution of the structure.
            - int: The number of particles in the structure.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("cif", cif_file)
    resolution, num_particles = get_em_stats(cif_file)

    pdb_id = cif_file.split("/")[-1][:-4]
    model = structure[0]
    ligands = {}

    for chain in model:
        chain_id = chain.get_id()

        for residue in chain:
            if is_studied_ligand(residue):
                ligand_coords = []
                ligand_name = f"{pdb_id}_{chain_id}_{residue.get_id()[1]}_{residue.get_id()[0][2:]}"
                for atom in residue:
                    ligand_coords.append((atom.element, atom.get_coord()))

                ligands[ligand_name] = ligand_coords

    return ligands, resolution, num_particles

def get_em_stats(cif_file):
    """
    Parses a CIF file and returns the resolution and number of particles
    for the EM reconstruction.

    Args:
        cif_file (str): Path to the input CIF file.

    Returns:
        tuple: A tuple containing the resolution (float) and number of particles (int).
    """
    resolution = None
    num_particles = None

    for line in open(cif_file):
        if line.startswith("_em_3d_reconstruction.resolution "):
            try:
                resolution = round(float(line.split()[1]), 1)
            except Exception:
                resolution = None
        if line.startswith("_em_3d_reconstruction.num_particles "):
            try:
                num_particles = int(line.split()[1])
            except Exception:
                num_particles = None

    if resolution is not None:
        if resolution > 4.0:
            resolution = 4.0
        elif resolution < 1.0:
            resolution = 1.0

    return resolution, num_particles

def is_studied_ligand(residue):
    """
    Determines whether a given residue is a studied ligand.

    Args:
        residue (Bio.PDB.Residue): The residue to check.

    Returns:
        bool: True if the residue is a studied ligand, False otherwise.
    """
    return residue.get_id()[0].startswith("H_")

# ==== Creating a sphere ==== #
# Borrowed from https://stackoverflow.com/a/46626448

def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (x_i / semisize) ** 2

    return arr <= 1.0

