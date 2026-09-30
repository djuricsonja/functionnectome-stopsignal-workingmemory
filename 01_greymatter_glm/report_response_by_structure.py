"""
The group response inside each anatomical structure.

IN   <group folder>            argument 1, e.g. stopsignal/group_all
     <label map>.nii.gz        argument 2, e.g. aseg_labels_50pc.nii.gz, made by
                               build_group_aseg_counts.py arm1 and write_group_aseg_labels.py arm1 0.5
                               (03_greymatter_mask)
     <label map>.csv           label,name table beside the label map
     [threshold]               argument 3, optional t cut-off
     <group folder>/<contrast>/{spmT_0001.nii, spmT_0002.nii, mask.nii}
                               made by group_level_stop_signal.m arm1

OUT  logs/response_by_structure_<group folder name>.csv
     printed table per contrast and direction
"""

import csv
import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

DIRECTIONS = [("pos", "spmT_0001.nii"), ("neg", "spmT_0002.nii")]
MASK_FILE = "mask.nii"

ATLAS_LABEL_COLUMN = "label"
ATLAS_NAME_COLUMN = "name"
SIDE_PREFIXES = ("Left-", "Right-")
PERCENTILES = [50, 95]
MIN_VOXELS = 1
# ====================================================================


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


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("give the group folder and the label map")
    groupDir = sys.argv[1]
    atlasImage = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if not os.path.isdir(groupDir):
        sys.exit(f"not found: {groupDir}")
    contrasts = sorted(d for d in os.listdir(groupDir)
                       if os.path.isdir(os.path.join(groupDir, d)))
    if not contrasts:
        sys.exit(f"no contrast folders in {groupDir}")

    atlas = nib.load(atlasImage)
    atlasData = np.asanyarray(atlas.dataobj)
    csvPath, atlasNames = read_atlas_names(atlasImage)

    print(f"group folder: {groupDir}")
    print(f"label map:    {atlasImage}")
    print(f"label table:  {csvPath}")
    print(f"threshold:    {threshold if threshold is not None else 'none given'}")
    print(f"contrasts:    {', '.join(contrasts)}\n")

    present = [int(v) for v in np.unique(atlasData) if v != 0]
    unnamed = [v for v in present if v not in atlasNames]
    if unnamed:
        sys.exit(f"{len(unnamed)} label values have no name in {csvPath}: {unnamed[:10]}")

    groups = {}
    for v in present:
        key = merge_side(atlasNames[v])
        groups.setdefault(key, np.zeros(atlasData.shape, dtype=bool))
        groups[key] |= atlasData == v

    rows = []
    for contrast in contrasts:
        d = os.path.join(groupDir, contrast)
        maskPath = os.path.join(d, MASK_FILE)
        if not os.path.isfile(maskPath):
            print(f"{contrast}: no {MASK_FILE}, skipping\n")
            continue
        maskImg = nib.load(maskPath)
        if maskImg.shape[:3] != atlasData.shape or not np.allclose(maskImg.affine, atlas.affine):
            sys.exit(f"ERROR: {maskPath} is not on the same grid as the label map")
        mask = np.asanyarray(maskImg.dataobj) > 0

        for direction, tFile in DIRECTIONS:
            tPath = os.path.join(d, tFile)
            if not os.path.isfile(tPath):
                print(f"{contrast} {direction}: no {tFile}, skipping")
                continue
            t = np.asanyarray(nib.load(tPath).dataobj, dtype=np.float64)
            t = np.where(np.isfinite(t), t, -np.inf)

            header = (f"{'structure':<26}{'tested':>9}{'of struct':>11}"
                      f"{'median t':>10}{'p95 t':>9}{'max t':>9}")
            if threshold is not None:
                header += f"{'above':>8}{'of struct':>11}"
            print(f"--- {contrast}  {direction} ---")
            print(header)

            out = []
            for key in groups:
                sel = np.logical_and(groups[key], mask)
                n = int(sel.sum())
                if n < MIN_VOXELS:
                    continue
                vals = t[sel]
                total = int(groups[key].sum())
                med, p95 = np.percentile(vals, PERCENTILES)
                mx = float(vals.max())
                above = int((vals >= threshold).sum()) if threshold is not None else 0
                out.append((key, n, total, med, p95, mx, above))
            out.sort(key=lambda r: -r[5])
            for key, n, total, med, p95, mx, above in out:
                line = (f"{key:<26}{n:>9,}{100 * n / total:>10.1f}%"
                        f"{med:>10.2f}{p95:>9.2f}{mx:>9.2f}")
                if threshold is not None:
                    line += f"{above:>8,}{100 * above / n:>10.1f}%"
                print(line)
                rows.append([contrast, direction, key, n, total,
                             round(med, 4), round(p95, 4), round(mx, 4), above])
            print()

    os.makedirs(LOG_DIR, exist_ok=True)
    stem = os.path.basename(os.path.normpath(groupDir))
    outCsv = os.path.join(LOG_DIR, f"response_by_structure_{stem}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["contrast", "direction", "structure", "voxels_tested",
                      "voxels_in_structure", "median_t", "p95_t", "max_t",
                      "voxels_above_threshold"])
        wtr.writerows(rows)
    print(f"wrote {outCsv}")


if __name__ == "__main__":
    main()
