"""
Write the group white-matter mask from the group anatomical label map, by structure name.

IN   label_map                        optional, default stopsignal/masks/aseg/aseg_labels_50pc.nii.gz
                                      made by write_group_aseg_labels.py priors (03_greymatter_mask), .csv of the same stem beside it
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  stopsignal/masks/group_aseg/group_wm_mask.nii.gz
"""

import csv
import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
MASK_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "masks")
ATLAS_IMAGE = os.path.join(MASK_ROOT, "aseg", "aseg_labels_50pc.nii.gz")
PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
OUT_MASK = os.path.join(MASK_ROOT, "group_aseg", "group_wm_mask.nii.gz")

ATLAS_LABEL_COLUMN = "label"
ATLAS_NAME_COLUMN = "name"
SIDE_PREFIXES = ("Left-", "Right-")

WM_STRUCTURES = (
    "Cerebral-White-Matter",
    "Cerebellum-White-Matter",
    "CC_Posterior",
    "CC_Mid_Posterior",
    "CC_Central",
    "CC_Mid_Anterior",
    "CC_Anterior",
    "WM-hypointensities",
    "Optic-Chiasm",
)
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
    atlasPath = sys.argv[1] if len(sys.argv) > 1 else ATLAS_IMAGE
    for p in (atlasPath, PRIORS_TEMPLATE):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: not found: {p}")

    tpl = nib.load(PRIORS_TEMPLATE)
    atlas = nib.load(atlasPath)
    if atlas.shape[:3] != tpl.shape[:3] or not np.allclose(atlas.affine, tpl.affine):
        sys.exit("ERROR: the label map is not on the priors grid")
    atlasData = np.asanyarray(atlas.dataobj)
    brain = np.asanyarray(tpl.dataobj) > 0
    csvPath, names = read_atlas_names(atlasPath)

    present = [int(v) for v in np.unique(atlasData) if v != 0]
    unnamed = [v for v in present if v not in names]
    if unnamed:
        sys.exit(f"{len(unnamed)} label values have no name in {csvPath}: {unnamed[:10]}")

    byStructure = {}
    for v, n in names.items():
        byStructure.setdefault(merge_side(n), []).append(v)
    absent = [s for s in WM_STRUCTURES if s not in byStructure]
    if absent:
        sys.exit(f"structures with no label in {csvPath}: {absent}")

    print(f"label map: {atlasPath}")
    print(f"table:     {csvPath}")
    print(f"template:  {PRIORS_TEMPLATE}, {int(brain.sum()):,} voxels\n")

    mask = np.zeros(atlasData.shape, dtype=bool)
    print(f"{'structure':<26}{'labels':<22}{'voxels':>9}{'in template':>13}")
    for s in WM_STRUCTURES:
        vals = sorted(byStructure[s])
        sel = np.isin(atlasData, vals)
        inTpl = np.logical_and(sel, brain)
        mask |= inTpl
        print(f"{s:<26}{','.join(str(v) for v in vals):<22}{int(sel.sum()):>9,}{int(inTpl.sum()):>13,}")
    print(f"\nwhite-matter mask: {int(mask.sum()):,} voxels")

    os.makedirs(os.path.dirname(OUT_MASK), exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), tpl.affine), OUT_MASK)
    print(f"wrote {OUT_MASK}")


if __name__ == "__main__":
    main()
