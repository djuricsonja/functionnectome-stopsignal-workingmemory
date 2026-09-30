"""
Structure-by-structure comparison of two grey-matter masks.

IN   label_map                        stopsignal/masks/aseg/aseg_labels_50pc.nii.gz
                                      made by write_group_aseg_labels.py priors, .csv of the same stem beside it
     mask_A                           stopsignal/masks/group/group_gm_mask.nii.gz, made by write_group_gm_mask.py
     mask_B                           stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py

OUT  logs/compare_masks_<mask A stem>_<mask B stem>.csv
"""

import csv
import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

ATLAS_LABEL_COLUMN = "label"
ATLAS_NAME_COLUMN = "name"
SIDE_PREFIXES = ("Left-", "Right-")
UNNAMED_LABEL = "unnamed by the label map"
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


def stem(path):
    base = os.path.basename(path)
    for suffix in (".nii.gz", ".nii"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def main():
    if len(sys.argv) != 4:
        sys.exit("give the label map and the two masks")
    atlasImage, maskAPath, maskBPath = sys.argv[1:4]

    atlas = nib.load(atlasImage)
    atlasData = np.asanyarray(atlas.dataobj)
    masks = []
    for p in (maskAPath, maskBPath):
        img = nib.load(p)
        if img.shape[:3] != atlasData.shape or not np.allclose(img.affine, atlas.affine):
            sys.exit(f"ERROR: {p} is not on the same grid as the label map")
        masks.append(np.asanyarray(img.dataobj) > 0)
    A, B = masks

    csvPath, atlasNames = read_atlas_names(atlasImage)
    present = [int(v) for v in np.unique(atlasData) if v != 0]
    unnamed = [v for v in present if v not in atlasNames]
    if unnamed:
        sys.exit(f"{len(unnamed)} label values have no name in {csvPath}: {unnamed[:10]}")

    groups = {}
    for v in present:
        key = merge_side(atlasNames[v])
        groups.setdefault(key, np.zeros(atlasData.shape, dtype=bool))
        groups[key] |= atlasData == v
    named = np.zeros(atlasData.shape, dtype=bool)
    for g in groups.values():
        named |= g
    groups[UNNAMED_LABEL] = ~named

    print(f"label map: {atlasImage}")
    print(f"mask A:    {maskAPath}   {int(A.sum()):,} voxels")
    print(f"mask B:    {maskBPath}   {int(B.sum()):,} voxels")
    print(f"           B adds {int(np.logical_and(~A, B).sum()):,}, "
          f"drops {int(np.logical_and(A, ~B).sum()):,}\n")

    rows = []
    for key, sel in groups.items():
        total = int(sel.sum())
        inA = int(np.logical_and(sel, A).sum())
        inB = int(np.logical_and(sel, B).sum())
        added = int(np.logical_and(sel, np.logical_and(~A, B)).sum())
        dropped = int(np.logical_and(sel, np.logical_and(A, ~B)).sum())
        if inA or inB:
            rows.append([key, total, inA, inB, added, dropped])
    rows.sort(key=lambda r: -(r[4] + r[5]))

    print(f"{'structure':<28}{'voxels':>9}{'in A':>9}{'in B':>9}{'added':>9}{'dropped':>9}")
    for key, total, inA, inB, added, dropped in rows:
        print(f"{key:<28}{total:>9,}{inA:>9,}{inB:>9,}{added:>9,}{dropped:>9,}")

    print(f"\nadded   {sum(r[4] for r in rows):,} over all structures")
    print(f"dropped {sum(r[5] for r in rows):,} over all structures")

    os.makedirs(LOG_DIR, exist_ok=True)
    outCsv = os.path.join(LOG_DIR, f"compare_masks_{stem(maskAPath)}_{stem(maskBPath)}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["structure", "voxels", "in_mask_A", "in_mask_B", "added_by_B",
                      "dropped_by_B"])
        wtr.writerows(rows)
    print(f"wrote {outCsv}")


if __name__ == "__main__":
    main()
