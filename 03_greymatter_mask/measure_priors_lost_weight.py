"""
Share of each sampled voxel's fibre probability inside a grey-matter mask, and where the rest lands.

IN   proj | asso | wb | comm          priors set
     mask                             stopsignal/masks/group/group_gm_mask.nii.gz, made by write_group_gm_mask.py,
                                      or stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py
     n_sample                         optional, default 500
     label_map                        optional, default stopsignal/masks/aseg/aseg_labels_50pc.nii.gz
                                      made by write_group_aseg_labels.py priors, .csv of the same stem beside it
     priors/<set file>.h5             Functionnectome priors, file names in CONFIG
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  logs/priors_lost_weight_<set>_<mask folder>_<mask stem>.csv
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

ATLAS_IMAGE = os.path.join(PROJECT_ROOT, "stopsignal", "masks", "aseg",
                           "aseg_labels_50pc.nii.gz")
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
OTHER_LABEL = "outside_the_map"
# ====================================================================


def summarise(name, values, unit=""):
    v = np.asarray(values, dtype=float)
    qs = np.percentile(v, PERCENTILES)
    cells = "  ".join(f"p{p}={q:,.2f}" for p, q in zip(PERCENTILES, qs))
    print(f"  {name:<34} min={v.min():,.2f}  {cells}  max={v.max():,.2f} {unit}")


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
    if len(sys.argv) not in (3, 4, 5):
        sys.exit("give the priors set and the mask")
    setLabel, maskPath = sys.argv[1:3]
    if setLabel not in PRIORS_FILE:
        sys.exit(f"unknown set {setLabel!r}; give one of " + ", ".join(sorted(PRIORS_FILE)))
    n_sample = int(sys.argv[3]) if len(sys.argv) > 3 else N_SAMPLE
    atlasImage = sys.argv[4] if len(sys.argv) > 4 else ATLAS_IMAGE

    h5path = os.path.join(PRIORS_DIR, PRIORS_FILE[setLabel])
    outCsv = os.path.join(LOG_DIR, f"priors_lost_weight_{setLabel}_{stem(maskPath)}.csv")

    tpl = nib.load(TEMPLATE)
    gm = nib.load(maskPath)
    atlas = nib.load(atlasImage)
    tplData = np.asanyarray(tpl.dataobj)
    gmData = np.asanyarray(gm.dataobj) > 0
    atlasData = np.asanyarray(atlas.dataobj)

    print(f"set:          {setLabel}")
    print(f"priors:       {h5path}")
    print(f"grey matter:  {maskPath}")
    print(f"template:     {TEMPLATE}")
    print(f"label map:    {atlasImage}")
    for name, img, data in (("grey-matter mask", gm, gmData), ("label map", atlas, atlasData)):
        if data.shape != tplData.shape:
            sys.exit(f"shape mismatch: {name} {data.shape} vs template {tplData.shape}")
        if not np.allclose(img.affine, tpl.affine):
            sys.exit(f"affine mismatch between the {name} and the priors template")
    print(f"grid:         {tplData.shape}, template voxels {int((tplData > 0).sum()):,}, "
          f"mask voxels {int(gmData.sum()):,}")

    csvPath, atlasNames = read_atlas_names(atlasImage)
    present = [int(v) for v in np.unique(atlasData) if v != 0]
    unnamed = [v for v in present if v not in atlasNames]
    if unnamed:
        sys.exit(f"{len(unnamed)} label values in the map have no name in {csvPath}: "
                 f"{unnamed[:10]}")
    print(f"label table:  {csvPath}  ({len(present)} labels present in the map)")

    groups = {}
    for v in present:
        key = merge_side(atlasNames[v])
        groups.setdefault(key, np.zeros(tplData.shape, dtype=bool))
        groups[key] |= atlasData == v
    named = np.zeros(tplData.shape, dtype=bool)
    for g in groups.values():
        named |= g
    groups[UNNAMED_LABEL] = ~named
    groupNames = sorted(groups)
    print(f"structures:   {len(groupNames)} after adding left and right together\n")

    idx = np.argwhere(tplData > 0)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(idx), size=min(n_sample, len(idx)), replace=False)
    sample = idx[pick]
    print(f"sampling:     {len(sample):,} template voxels, seed {SEED}\n")

    fields = (["set", "i", "j", "k", "w_total", "w_in_mask", "w_lost"]
              + [f"lost_{g}" for g in groupNames] + [f"lost_{OTHER_LABEL}"])
    rows = []
    empty = 0
    with h5py.File(h5path, "r") as h5:
        grp = h5[H5_GROUP]
        for n, (i, j, k) in enumerate(sample, 1):
            key = KEY.format(i, j, k)
            if key not in grp:
                sys.exit(f"no map stored for template voxel {i},{j},{k}")
            w = np.asarray(grp[key][:], dtype=np.float64)
            total = float(w.sum())
            if total <= 0:
                empty += 1
                continue
            inMask = float(w[gmData].sum())
            outside = w.copy()
            outside[gmData] = 0.0
            row = {"set": setLabel, "i": int(i), "j": int(j), "k": int(k),
                   "w_total": round(total, 6), "w_in_mask": round(inMask, 6),
                   "w_lost": round(total - inMask, 6)}
            namedSum = 0.0
            for g in groupNames:
                v = float(outside[groups[g]].sum())
                namedSum += v
                row[f"lost_{g}"] = round(v, 6)
            row[f"lost_{OTHER_LABEL}"] = round(total - inMask - namedSum, 6)
            rows.append(row)
            if n % 100 == 0:
                print(f"  {n}/{len(sample)} sampled")

    if not rows:
        sys.exit("every sampled voxel was empty; nothing to summarise")

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields)
        wtr.writeheader()
        wtr.writerows(rows)

    tot = np.array([r["w_total"] for r in rows], dtype=float)
    print(f"\n{setLabel}: {len(rows):,} of {len(sample):,} sampled voxels carry probability; "
          f"{empty:,} carry none")
    summarise("total probability per voxel", tot)
    summarise("share kept, inside the mask",
              [100 * r["w_in_mask"] / r["w_total"] for r in rows], "%")
    summarise("share lost, outside the mask",
              [100 * r["w_lost"] / r["w_total"] for r in rows], "%")

    print("\nwhere the lost probability lands, as a share of each voxel's total")
    order = sorted(groupNames + [OTHER_LABEL],
                   key=lambda g: -float(np.median([100 * r[f"lost_{g}"] / r["w_total"]
                                                   for r in rows])))
    for g in order:
        share = [100 * r[f"lost_{g}"] / r["w_total"] for r in rows]
        if max(share) <= 0:
            continue
        summarise(g, share, "%")

    print(f"\nper-voxel values: {outCsv}")


if __name__ == "__main__":
    main()
