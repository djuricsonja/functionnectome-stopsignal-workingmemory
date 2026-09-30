"""
Combined share of each sampled voxel's in-mask fibre weight supplied by a group of structures.

IN   source_csv                       logs/priors_source_by_structure_<set>_<mask folder>_<mask stem>.csv
                                      made by measure_priors_source_by_structure.py

OUT  printed median and mean over the sampled voxels
"""

import csv
import sys

import numpy as np

# ============================== CONFIG ==============================
STRUCTURES = ("Brain-Stem", "VentralDC", "Thalamus")
COLUMN_PREFIX = "w_"
IN_MASK_COLUMN = "w_in_mask"
# ====================================================================


def main():
    if len(sys.argv) != 2:
        sys.exit("give the priors_source_by_structure CSV")
    path = sys.argv[1]

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"no rows in {path}")

    columns = [COLUMN_PREFIX + s for s in STRUCTURES]
    missing = [c for c in columns + [IN_MASK_COLUMN] if c not in rows[0]]
    if missing:
        sys.exit(f"columns not in {path}: {', '.join(missing)}")

    inMask = np.array([float(r[IN_MASK_COLUMN]) for r in rows])
    combined = np.zeros(len(rows))
    for c in columns:
        combined += np.array([float(r[c]) for r in rows])
    share = 100 * combined / inMask

    print(f"file:        {path}")
    print(f"structures:  {', '.join(STRUCTURES)}")
    print(f"voxels:      {len(rows)}")
    print(f"combined share of the in-mask weight: median {np.median(share):.1f}%, "
          f"mean {share.mean():.1f}%")


if __name__ == "__main__":
    main()
