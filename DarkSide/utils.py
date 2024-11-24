import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
import mendeleev
from functools import lru_cache
from scipy.spatial import KDTree
import pyopencl as cl
import mrcfile

# ==== Handling MRC file ==== #
def load_map(mrc_path):
    with mrcfile.open(mrc_path) as mrc:
        assert mrc.is_volume()

        cell_a = mrc.header.cella[["x","y","z"]].view(("f4", 3))
        origin = np.array([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart]).astype("f4")

        order = (3 - mrc.header.maps, 3 - mrc.header.mapr, 3 - mrc.header.mapc) # NOTE: this might have ZYX ordering
        map = np.asarray(mrc.data, dtype="f4")
        map = np.moveaxis(a=map, source=order, destination=(2, 1, 0))

        scale = map.shape / cell_a # NOTE: this might have ZYX ordering

        return map, scale, origin

def cut_out_map(map, scale, offset, size):
    size = np.ceil(size * scale).astype(int)
    offset = np.floor(offset * scale).astype(int)

    cutout = map[offset[0]:offset[0]+size[0],offset[1]:offset[1]+size[1],offset[2]:offset[2]+size[2]]

    return cutout

# ==== OpenCL min distance ==== #
def init_cl(platform = 0):
    cl_platform = cl.get_platforms()[platform]

    devices = cl_platform.get_devices(cl.device_type.GPU)
    if len(devices) == 0:
        devices = cl_platform.get_devices(device_type=cl.device_type.CPU)

    cl_ctx = cl.Context(devices=devices)
    cl_queue = cl.CommandQueue(cl_ctx)

    prg = cl.Program(cl_ctx, """
    __kernel void calc_min_distance(__global float* min_distances, const int num_atoms, __constant const float* atom_center, __constant const float* atom_radius) {
        const int x = get_global_id(0);
        const int y = get_global_id(1);
        const int z = get_global_id(2);
        const int width  = get_global_size(0);
        const int height = get_global_size(1);
        const int depth  = get_global_size(2);

        const int idx = x * height * depth + y * depth + z;

        const float3 position = (float3)(x, y, z);

        float min_distance = FLT_MAX;
        for (int i = 0; i < num_atoms; i++) {
            const float3 center = (float3)(atom_center[i*3+0], atom_center[i*3+1], atom_center[i*3+2]);
            const float dist = length(position - center) - atom_radius[i];
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
        min_distances[idx] = min_distance;
    }
    """).build()

    def atoms_to_cl_args(ctx, atoms):
        num_atoms = len(atoms)
        atom_center_np = np.empty(shape=(num_atoms, 3), dtype=np.float32)
        atom_radius_np = np.empty(shape=(num_atoms,), dtype=np.float32)
        for idx, atom in enumerate(atoms):
            atom_center_np[idx] = atom[0].astype(np.float32)
            atom_radius_np[idx] = np.mean(atom[1]).astype(np.float32)
        atom_center_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=atom_center_np)
        atom_radius_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=atom_radius_np)
        return np.int32(num_atoms), atom_center_buf, atom_radius_buf

    def calc_min_distance(atoms, shape):
        distance_np = np.empty(shape=shape, dtype=np.float32)
        distance_cl = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, distance_np.nbytes)

        prg.calc_min_distance(cl_queue, shape, None, distance_cl, *atoms_to_cl_args(cl_ctx, atoms))

        cl.enqueue_copy(cl_queue, distance_np, distance_cl)

        return distance_np

    return cl_ctx, calc_min_distance

# ==== Neighbors lookup ==== #
def get_lookup(ligands, origin, scale):
    LOOKUP_POINTS_PER_ATOM = 2
    ligands_pc = []
    ligands_names = []
    ligands_atoms = []
    for ligand_name in ligands:
        for element_name, position in ligands[ligand_name]:
            center = position - origin
            radius = get_atomic_radius(element_name)
            ligands_pc.append((center - radius) * scale)
            ligands_pc.append((center + radius) * scale)
            ligands_atoms.append((center * scale, radius * scale))
            ligands_names.append(ligand_name)

    lookup = KDTree(ligands_pc)

    def lookup_neighbors(offset, size, exclude):
        # ideally the KDTree would implement query_rectangle([offset, offset + size])
        center = offset + size / 2
        radius = np.ceil((size @ size) ** 0.5 * 0.5)

        ligand_atoms_idxs = [idx for idx, ligand_name in enumerate(ligands_names) if ligand_name == exclude]
        neighbor_atoms_idxs = lookup.query_ball_point(center * scale, (radius * scale).mean())
        neighbor_atoms_idxs = [i // LOOKUP_POINTS_PER_ATOM for i in neighbor_atoms_idxs]
        neighbor_atoms_idxs = filter(lambda idx: not idx in ligand_atoms_idxs, neighbor_atoms_idxs)
        neighbor_atoms_idxs = list(set(neighbor_atoms_idxs))

        neighbor_atoms = [(ligands_atoms[idx][0] - offset*scale, ligands_atoms[idx][1]) for idx in neighbor_atoms_idxs]
        return neighbor_atoms

    return lookup_neighbors

# ==== Handling atoms ==== #
def ligand_to_atoms(ligand, offset, scale=np.ones(shape=[3]), origin=np.zeros(shape=[3])):
    return [((position - offset) * scale - origin, get_atomic_radius(element_name) * scale) for element_name, position in ligand]

def get_mask(atoms, shape):
    bb_min = np.floor(np.array([center - radius for center, radius in atoms]).min(axis=0)).astype(int)
    bb_max = np.ceil(np.array([center + radius for center, radius in atoms]).max(axis=0)).astype(int)

    mask = np.zeros(shape=bb_max - bb_min)
    for center, radius in atoms:
        mask = np.logical_or(mask, sphere(mask.shape, radius.mean(), center - bb_min))

    return mask[-bb_min[0]:shape[0]-bb_min[0], -bb_min[1]:shape[1]-bb_min[1], -bb_min[2]:shape[2]-bb_min[2]]

def binarize(map, threshold):
    binary = map.copy()
    binary[map >= threshold] = 1
    binary[map <  threshold] = 0
    return binary

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
    ligands = {"not_studied": []}

    for chain in model:
        chain_id = chain.get_id()

        for residue in chain:
            ligand_coords = []
            for atom in residue:
                ligand_coords.append((atom.element, atom.get_coord()))

            if is_studied_ligand(residue):
                ligand_name = f"{pdb_id}_{chain_id}_{residue.get_id()[1]}_{residue.get_id()[0][2:]}"
                ligands[ligand_name] = ligand_coords
            else:
                ligands["not_studied"] += ligand_coords

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

