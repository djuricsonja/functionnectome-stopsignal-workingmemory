"""
Share of each structure in the mask that each participant's scan did not sample.

IN   mask                             stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py (03_greymatter_mask)
     label_map                        stopsignal/masks/aseg/aseg_labels_50pc.nii.gz
                                      made by write_group_aseg_labels.py priors (03_greymatter_mask), .csv of the same stem beside it
     min_missing                      optional, default 100
     stopsignal/functionnectome/input/<sub>_support.nii.gz   made by build_funtome_input.py (02_template_and_route)
     subjects_stopsignal_all.txt   one participant ID per line, in this folder

OUT  logs/missing_by_structure_<mask folder>_<mask stem>.csv
"""

import csv
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
SUPPORT_DIR = os.path.join(PROJECT_ROOT, "stopsignal", "functionnectome", "input")
SUPPORT_SUFFIX = "_support.nii.gz"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FILE = os.path.join(SCRIPT_DIR, "subjects_stopsignal_all.txt")

ATLAS_LABEL_COLUMN = "label"
ATLAS_NAME_COLUMN = "name"
SIDE_PREFIXES = ("Left-", "Right-")
UNNAMED_LABEL = "unnamed"

MIN_MISSING = 100      # subjects with fewer uncovered voxels are not listed
TOP_STRUCTURES = 4
TOP_PIECES = 3
SUMMARY_SHARE = 0.10   # a structure counts as badly hit above this share
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
    full = os.path.abspath(path)
    base = os.path.basename(full)
    for suffix in (".nii.gz", ".nii"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.basename(os.path.dirname(full)) + "_" + base


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("give the mask, the label map, and optionally the minimum missing count")
    maskPath, atlasPath = sys.argv[1:3]
    minMissing = int(sys.argv[3]) if len(sys.argv) > 3 else MIN_MISSING

    maskImg = nib.load(maskPath)
    mask = np.asanyarray(maskImg.dataobj) > 0
    atlas = nib.load(atlasPath)
    atlasData = np.asanyarray(atlas.dataobj)
    if atlasData.shape != mask.shape or not np.allclose(atlas.affine, maskImg.affine):
        sys.exit("ERROR: the label map is not on the same grid as the mask")
    csvPath, atlasNames = read_atlas_names(atlasPath)

    subs = [l.strip() for l in open(SUBJECT_FILE) if l.strip()]

    print(f"mask:       {maskPath}   {int(mask.sum()):,} voxels")
    print(f"label map:  {atlasPath}")
    print(f"listing subjects with at least {minMissing} uncovered mask voxels\n")

    groups = {}
    for v in [int(x) for x in np.unique(atlasData) if x != 0]:
        key = merge_side(atlasNames[v]) if v in atlasNames else f"label-{v}"
        groups.setdefault(key, np.zeros(mask.shape, dtype=bool))
        groups[key] |= atlasData == v
    named = np.zeros(mask.shape, dtype=bool)
    for g in groups.values():
        named |= g
    groups[UNNAMED_LABEL] = ~named
    inMask = {k: np.logical_and(g, mask) for k, g in groups.items()}
    sizes = {k: int(g.sum()) for k, g in inMask.items()}

    rows = []
    hit = {k: 0 for k in inMask}
    for sub in subs:
        f = os.path.join(SUPPORT_DIR, sub + SUPPORT_SUFFIX)
        if not os.path.isfile(f):
            continue
        support = np.asanyarray(nib.load(f).dataobj) > 0
        miss = np.logical_and(mask, ~support)
        n = int(miss.sum())
        if n < minMissing:
            continue
        lab, pieces = ndimage.label(miss)
        pieceSizes = sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True)

        perStructure = []
        for k, g in inMask.items():
            m = int(np.logical_and(g, miss).sum())
            if m:
                share = m / sizes[k] if sizes[k] else 0.0
                perStructure.append((share, m, k))
                if share >= SUMMARY_SHARE:
                    hit[k] += 1
        perStructure.sort(reverse=True)

        top = "  ".join(f"{k} {100 * share:.0f}% ({m:,})"
                        for share, m, k in perStructure[:TOP_STRUCTURES])
        print(f"{sub}   missing {n:,} in {pieces} pieces, largest "
              f"{', '.join(f'{p:,}' for p in pieceSizes[:TOP_PIECES])}"
              f"   ({100 * pieceSizes[0] / n:.0f}% in one)")
        print(f"              {top}")
        for share, m, k in perStructure:
            rows.append([sub, n, pieces, pieceSizes[0], k, sizes[k], m, round(share, 4)])

    print(f"\nstructures with more than {SUMMARY_SHARE:.0%} missing, "
          f"counted over the listed subjects")
    for k in sorted(hit, key=lambda k: -hit[k]):
        if hit[k]:
            print(f"   {k:<26}{hit[k]:>4} subjects   (structure holds "
                  f"{sizes[k]:,} mask voxels)")

    os.makedirs(LOG_DIR, exist_ok=True)
    outCsv = os.path.join(LOG_DIR, f"missing_by_structure_{stem(maskPath)}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["subject", "missing_total", "pieces", "largest_piece", "structure",
                      "structure_voxels_in_mask", "missing_in_structure", "share_missing"])
        wtr.writerows(rows)
    print(f"\nwrote {outCsv}")


if __name__ == "__main__":
    main()
