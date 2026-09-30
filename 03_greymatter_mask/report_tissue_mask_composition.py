"""
Structure-by-structure composition of the tissue-based grey-matter mask.

IN   label_map                        optional, default stopsignal/masks/aseg/aseg_labels_50pc.nii.gz
                                      made by write_group_aseg_labels.py priors, .csv of the same stem beside it
     mask                             optional, default stopsignal/masks/group/group_gm_mask.nii.gz
                                      made by write_group_gm_mask.py
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  logs/mask_composition_<label map stem>.csv
"""

import csv
import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
ATLAS_IMAGE = os.path.join(PROJECT_ROOT, "stopsignal", "masks", "aseg",
                           "aseg_labels_50pc.nii.gz")
MASK_IMAGE = os.path.join(PROJECT_ROOT, "stopsignal", "masks", "group",
                          "group_gm_mask.nii.gz")
TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
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


def main():
    atlasImage = sys.argv[1] if len(sys.argv) > 1 else ATLAS_IMAGE
    maskImage = sys.argv[2] if len(sys.argv) > 2 else MASK_IMAGE

    tpl = nib.load(TEMPLATE)
    atlas = nib.load(atlasImage)
    mask = nib.load(maskImage)
    for name, img in (("label map", atlas), ("mask", mask)):
        if img.shape[:3] != tpl.shape[:3] or not np.allclose(img.affine, tpl.affine):
            sys.exit(f"ERROR: the {name} is not on the priors grid")

    atlasData = np.asanyarray(atlas.dataobj)
    maskData = np.asanyarray(mask.dataobj) > 0
    brain = np.asanyarray(tpl.dataobj) > 0
    csvPath, atlasNames = read_atlas_names(atlasImage)

    print(f"label map: {atlasImage}")
    print(f"table:     {csvPath}")
    print(f"mask:      {maskImage}")
    print(f"grid:      {atlasData.shape}, priors brain {int(brain.sum()):,} voxels, "
          f"mask {int(maskData.sum()):,} voxels\n")

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
    groups[UNNAMED_LABEL] = np.logical_and(~named, brain)

    nMask = int(maskData.sum())
    rows = []
    for key, sel in groups.items():
        total = int(sel.sum())
        kept = int(np.logical_and(sel, maskData).sum())
        rows.append([key, total, kept,
                     round(100 * kept / total, 2) if total else 0.0,
                     round(100 * kept / nMask, 2) if nMask else 0.0])
    rows.sort(key=lambda r: -r[2])

    print(f"{'structure':<28}{'voxels':>10}{'in mask':>10}{'of structure':>14}{'of mask':>10}")
    for key, total, kept, ofStruct, ofMask in rows:
        print(f"{key:<28}{total:>10,}{kept:>10,}{ofStruct:>13.1f}%{ofMask:>9.1f}%")

    accounted = sum(r[2] for r in rows)
    print(f"\nmask voxels accounted for: {accounted:,} of {nMask:,}")

    os.makedirs(LOG_DIR, exist_ok=True)
    stem = os.path.basename(atlasImage).split(".nii")[0]
    outCsv = os.path.join(LOG_DIR, f"mask_composition_{stem}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["structure", "voxels", "voxels_in_mask",
                      "percent_of_structure", "percent_of_mask"])
        wtr.writerows(rows)
    print(f"wrote {outCsv}")


if __name__ == "__main__":
    main()
