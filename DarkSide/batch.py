from blobify import *
import argparse
import yaml
from tqdm import tqdm
import os
import gevent
from gevent.pool import Pool
from gevent import monkey
monkey.patch_all()
import requests
import subprocess

"""
Example yaml input
==================
data:
  - emd: 0199
    thr: 0.005
    pdb: 6hcy
    res: 3.1
"""

parser = argparse.ArgumentParser()
parser.add_argument("yaml", help="path to data yaml file")
parser.add_argument("-v", "--verbose", action="store_true", help="enable logging")
parser.add_argument("-c", "--clean-up", action="store_true", help="remove downloaded files")
parser.add_argument("-j", "--jobs", default=1, type=int, help="workers pool size")
args = parser.parse_args()

log = print if args.verbose else lambda x: x

pool = Pool(size=args.jobs)

log("Loading yaml")
with open(args.yaml) as input:
    data = yaml.safe_load(input)["data"]

jobs = []
lookup = dict()
done = set()
dependencies = dict()
for entry in data:
    emd = entry["emd"]
    pdb = entry["pdb"]
    resolution = entry["res"]
    threshold = entry["thr"]

    emd_url = f"https://files.rcsb.org/pub/emdb/structures/EMD-{emd}/map/emd_{emd}.map.gz"
    pdb_url = f"https://files.rcsb.org/download/{pdb.upper()}.cif"

    emd_path = f"data/emd_{emd}.map.gz"
    pdb_path = f"data/{pdb.upper()}.cif"

    to_schedule = list()
    if not os.path.isfile(emd_path):
        lookup[emd_url] = (entry, emd_url, pdb_url, emd_path, pdb_path, emd_path)
        to_schedule.append(emd_url)
        if not os.path.isfile(pdb_path):
            dependencies[emd_url] = pdb_url

    if not os.path.isfile(pdb_path):
        lookup[pdb_url] = (entry, emd_url, pdb_url, emd_path, pdb_path, pdb_path)
        to_schedule.append(pdb_url)
        if not os.path.isfile(emd_path):
            dependencies[pdb_url] = emd_url

    def schedule_cutting(pdb_path, emd_path, pdb, emd, resolution, threshold):
        jobs.append(pool.spawn(lambda args: subprocess.run(["python", "cutter.py", *args]), ["-o", f"data/NAME.npz", "-t", repr(threshold), "-i", pdb_path, emd_path]))
        log(f"Cutting EMD-{emd} (structure: {pdb.upper()}, resolution: {resolution}, threshold: {threshold})")

    def on_downloaded(job):
        response = job.value
        if response.url in done:
            return
        entry, emd_url, pdb_url, emd_path, pdb_path, path = lookup[response.url]
        emd = entry["emd"]
        pdb = entry["pdb"]
        resolution = entry["res"]
        threshold = entry["thr"]
        if response.status_code == 200:
            log(f"Writing {path}")
            with open(path, "wb") as file:
                file.write(response.content)
            done.add(response.url)
            schedule = False
            if response.url in dependencies:
                if dependencies[response.url] in done:
                    schedule = True
            else:
                schedule = True
            if schedule:
               schedule_cutting(pdb_path, emd_path, pdb, emd, resolution, threshold)
        else:
            print(f"Error while downloading {response.url}")

    for url in to_schedule:
        job = pool.spawn(lambda url: requests.get(url), url)
        log(f"Downloading {url}")
        job.link(on_downloaded)
        jobs.append(job)

    if len(to_schedule) == 0:
       schedule_cutting(pdb_path, emd_path, pdb, emd, resolution, threshold)

gevent.joinall(jobs)

