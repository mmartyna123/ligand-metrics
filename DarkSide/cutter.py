from blobify import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("cif", help="path to cif file")
parser.add_argument("mrc", help="path to mrc file")
parser.add_argument("-l", "--ligand", help="specific ligand to process")
parser.add_argument("-t", "--threshold", type=float, help="threshold the density map")
parser.add_argument("-i", "--intermediate", action="store_true", help="save the intermediate results")
parser.add_argument("-o", "--output", nargs="?", default="ours_NAME.npz", help="output pattern")
parser.add_argument("-v", "--verbose", action="store_true", help="enable logging")
args = parser.parse_args()

log = print if args.verbose else lambda x: x

log("Loading ligands")
ligands = load_ligands(args.cif)
log("Loading map")
map, scale, origin = load_map(args.mrc)
log("Creating lookup tree")
lookup = get_lookup(ligands, origin, scale)
log("Initializing OpenCL")
cl_ctx, calc_min_distance = init_cl()

if not args.ligand:
    to_process = [(name, ligand) for name, ligand in ligands.items() if name != "not_studied"]
else:
    to_process = [(args.ligand, ligands[args.ligand])]

log(f"Processing {len(to_process)} ligands")
to_process = tqdm(to_process, unit="ligand") if args.verbose else to_process

for name, ligand in to_process:
    offset, size = get_offset_and_size(ligand)
    cutout = cut_out_map(map, scale, offset, size).copy()

    positive_atoms = ligand_to_atoms(ligand, offset, scale, origin)
    negative_atoms = lookup(offset, size, exclude=name)

    distance_to_positive = calc_min_distance(positive_atoms, cutout.shape)
    distance_to_negative = calc_min_distance(negative_atoms, cutout.shape)

    cutout[distance_to_positive > distance_to_negative] = cutout.min()

    if args.threshold is None or args.intermediate:
        np.savez(args.output.replace("NAME", name), cutout)

    if args.threshold:
        cutout = binarize(cutout, args.threshold) * cutout

        np.savez(args.output.replace("NAME", f"{name}_T"), cutout)

