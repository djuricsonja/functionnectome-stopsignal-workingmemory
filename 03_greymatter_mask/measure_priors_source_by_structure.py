"""
Share of each sampled voxel's in-mask fibre weight supplied by each structure.

IN   proj | asso | wb | comm          priors set
     mask                             stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py
     label_map                        stopsignal/masks/aseg/aseg_labels_50pc.nii.gz
                                      made by write_group_aseg_labels.py priors, .csv of the same stem beside it
     n_sample                         optional, default 500
     priors/<set file>.h5             Functionnectome priors, file names in CONFIG
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  logs/priors_source_by_structure_<set>_<mask folder>_<mask stem>.csv
"""

import csv
import os
import sys

# Must precede the numpy import, as in functionnectome.py
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import h5py
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
PRIORS_DIR = os.path.join(PROJECT_ROOT, "priors")
TEMPLATE = os.path.join(PRIORS_DIR, "priors_template.nii.gz")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

ATLAS_LABEL_COLUMN = "label"
ATLAS_NAME_COLUMN = "name"
SIDE_PREFIXES = ("Left-", "Right-")
UNNAMED_LABEL = "unnamed"

PRIORS_FILE = {
    "proj": "priors_proj_proba_3T.h5",
    "asso": "priors_asso_proba_3T.h5",
    "comm": "priors_comm_proba_3T.h5",
    "wb": "priors_full_proba_3T_comp9_thr0p01.h5",
}
H5_GROUP = "tract_voxel"
KEY = "{}_{}_{}_vox"

N_SAMPLE = 500
SEED = 0
PERCENTILES = [5, 25, 50, 75, 95]
# ====================================================================


def summarise(name, values, unit="%"):
    v = np.asarray(values, dtype=float)
    qs = np.percentile(v, PERCENTILES)
    cells = "  ".join(f"p{p}={q:,.2f}" for p, q in zip(PERCENTILES, qs))
    print(f"  {name:<28} min={v.min():,.2f}  {cells}  max={v.max():,.2f} {unit}")


def read_atlas_names(imagePath):
    csvPath = imagePath
    for suffix in (".nii.gz", ".nii"):
        if csvPath.endswith(suffix):
            csvPath = csvPath[: -len(suffix)] + ".csv"
            break
    if not os.path.isfile(csvPath):
        sys.exit(f"no label table beside the label map: {csvPath}")
    names = {}
    with open(csvPath, newline="") as fh:
        for row in csv.DictReader(fh):
            names[int(row[ATLAS_LABEL_COLUMN])] = row[ATLAS_NAME_COLUMN]
    return csvPath, names


def merge_side(name):
    for p in SIDE_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def stem(path):
    full = os.path.abspath(path)
    base = os.path.basename(full)
    for suffix in (".nii.gz", ".nii"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.basename(os.path.dirname(full)) + "_" + base


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit("give the priors set, the mask and the label map")
    setLabel, maskPath, atlasPath = sys.argv[1:4]
    if setLabel not in PRIORS_FILE:
        sys.exit(f"unknown set {setLabel!r}; give one of " + ", ".join(sorted(PRIORS_FILE)))
    nSample = int(sys.argv[4]) if len(sys.argv) > 4 else N_SAMPLE

    h5path = os.path.join(PRIORS_DIR, PRIORS_FILE[setLabel])
    tpl = nib.load(TEMPLATE)
    tplData = np.asanyarray(tpl.dataobj)
    maskImg = nib.load(maskPath)
    mask = np.asanyarray(maskImg.dataobj) > 0
    atlas = nib.load(atlasPath)
    atlasData = np.asanyarray(atlas.dataobj)
    for name, img, data in (("mask", maskImg, mask), ("label map", atlas, atlasData)):
        if data.shape != tplData.shape or not np.allclose(img.affine, tpl.affine):
            sys.exit(f"ERROR: the {name} is not on the priors grid")

    csvPath, atlasNames = read_atlas_names(atlasPath)
    print(f"set:        {setLabel}")
    print(f"priors:     {h5path}")
    print(f"mask:       {maskPath}   {int(mask.sum()):,} voxels")
    print(f"label map:  {atlasPath}")

    groups = {}
    for v in [int(x) for x in np.unique(atlasData) if x != 0]:
        key = merge_side(atlasNames[v]) if v in atlasNames else f"label-{v}"
        groups.setdefault(key, np.zeros(tplData.shape, dtype=bool))
        groups[key] |= atlasData == v
    named = np.zeros(tplData.shape, dtype=bool)
    for g in groups.values():
        named |= g
    groups[UNNAMED_LABEL] = ~named
    inMask = {k: np.logical_and(g, mask) for k, g in groups.items()}
    inMask = {k: g for k, g in inMask.items() if g.any()}
    sizes = {k: int(g.sum()) for k, g in inMask.items()}
    groupNames = sorted(inMask)
    print(f"structures: {len(groupNames)} present inside the mask\n")

    idx = np.argwhere(tplData > 0)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(idx), size=min(nSample, len(idx)), replace=False)
    sample = idx[pick]
    print(f"sampling:   {len(sample):,} template voxels, seed {SEED}\n")

    fields = ["set", "i", "j", "k", "w_in_mask"] + [f"w_{g}" for g in groupNames]
    rows = []
    empty = 0
    with h5py.File(h5path, "r") as h5:
        grp = h5[H5_GROUP]
        for n, (i, j, k) in enumerate(sample, 1):
            key = KEY.format(i, j, k)
            if key not in grp:
                sys.exit(f"no map stored for template voxel {i},{j},{k}")
            w = np.asarray(grp[key][:], dtype=np.float64)
            inside = float(w[mask].sum())
            if inside <= 0:
                empty += 1
                continue
            row = {"set": setLabel, "i": int(i), "j": int(j), "k": int(k),
                   "w_in_mask": round(inside, 6)}
            for g in groupNames:
                row[f"w_{g}"] = round(float(w[inMask[g]].sum()), 6)
            rows.append(row)
            if n % 100 == 0:
                print(f"  {n}/{len(sample)} sampled")

    if not rows:
        sys.exit("no sampled voxel draws any weight from inside the mask")

    os.makedirs(LOG_DIR, exist_ok=True)
    outCsv = os.path.join(LOG_DIR,
                          f"priors_source_by_structure_{setLabel}_{stem(maskPath)}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"\n{setLabel}: {len(rows):,} of {len(sample):,} sampled voxels draw from the mask; "
          f"{empty:,} draw nothing")
    print("\nshare of the weight the projection uses, by structure")
    order = sorted(groupNames,
                   key=lambda g: -float(np.median([100 * r[f"w_{g}"] / r["w_in_mask"]
                                                   for r in rows])))
    for g in order:
        share = [100 * r[f"w_{g}"] / r["w_in_mask"] for r in rows]
        if max(share) <= 0:
            print(f"  {g:<28} no weight anywhere in the sample "
                  f"({sizes[g]:,} mask voxels)")
            continue
        summarise(f"{g} ({sizes[g]:,} vox)", share)

    print(f"\nper-voxel values: {outCsv}")


if __name__ == "__main__":
    main()
