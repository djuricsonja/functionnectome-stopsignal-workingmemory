"""
Mask voxels each participant's scan did not sample: count, connected pieces and z range.

IN   mask                             stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py (03_greymatter_mask)
     stopsignal/functionnectome/input/<sub>_support.nii.gz   made by build_funtome_input.py (02_template_and_route)
     subjects_stopsignal_all.txt   one participant ID per line, in this folder

OUT  logs/coverage_from_support_<mask folder>_<mask stem>.csv
     logs/subjects_excluded_coverage_<mask folder>_<mask stem>.txt   participants meeting MIN_BLOCK_VOXELS and MIN_LARGEST_SHARE
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

REFERENCE_SUBJECT_COLUMN = "subject"
REFERENCE_VALUE_COLUMN = "uncovered_any"

MIN_BLOCK_VOXELS = 500
MIN_LARGEST_SHARE = 0.90
SHOW = 0                      # subjects listed in the printed table, 0 for every one
# ====================================================================


def stem(path):
    full = os.path.abspath(path)
    base = os.path.basename(full)
    for suffix in (".nii.gz", ".nii"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.basename(os.path.dirname(full)) + "_" + base


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("give the mask, and optionally a reference CSV")
    maskPath = sys.argv[1]
    refPath = sys.argv[2] if len(sys.argv) > 2 else None

    for p in (maskPath, SUBJECT_FILE, SUPPORT_DIR):
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found: {p}")

    maskImg = nib.load(maskPath)
    mask = np.asanyarray(maskImg.dataobj) > 0
    subs = [l.strip() for l in open(SUBJECT_FILE) if l.strip()]

    print(f"mask:      {maskPath}   {int(mask.sum()):,} voxels")
    print(f"support:   {SUPPORT_DIR}/<sub>{SUPPORT_SUFFIX}")
    print(f"subjects:  {len(subs)}")
    print(f"rule:      gap larger than {MIN_BLOCK_VOXELS} voxels and more than "
          f"{MIN_LARGEST_SHARE:.0%} of it in one connected piece\n")

    rows = []
    skipped = []
    for sub in subs:
        f = os.path.join(SUPPORT_DIR, sub + SUPPORT_SUFFIX)
        if not os.path.isfile(f):
            skipped.append(sub)
            continue
        img = nib.load(f)
        if img.shape[:3] != mask.shape or not np.allclose(img.affine, maskImg.affine):
            sys.exit(f"ERROR: {f} is not on the same grid as the mask")
        support = np.asanyarray(img.dataobj) > 0
        miss = np.logical_and(mask, ~support)
        n = int(miss.sum())
        if n == 0:
            rows.append([sub, 0, 0, 0, 0.0, "", "", ""])
            continue
        lab, pieces = ndimage.label(miss)
        sizes = np.bincount(lab.ravel())[1:]
        biggest = int(sizes.max())
        share = biggest / n
        z = nib.affines.apply_affine(maskImg.affine, np.argwhere(miss))[:, 2]
        rows.append([sub, n, int(pieces), biggest, round(share, 4),
                     round(float(z.min()), 1), round(float(np.median(z)), 1),
                     round(float(z.max()), 1)])

    if skipped:
        print(f"no support file for {len(skipped)} subjects: {', '.join(skipped[:10])}\n")

    excluded = [r[0] for r in rows
                if r[1] > MIN_BLOCK_VOXELS and r[4] > MIN_LARGEST_SHARE]

    ranked = sorted([r for r in rows if r[1] > 0], key=lambda r: (-r[4], -r[1]))
    shown = ranked if SHOW <= 0 else ranked[:SHOW]
    print("sorted by the share in one piece, which is what the rule turns on\n")
    print(f"{'subject':<12}{'missing':>9}{'pieces':>8}{'largest':>9}{'share':>8}"
          f"{'z min':>8}{'z med':>8}{'z max':>8}{'over size':>11}   excluded")
    prev = None
    for r in shown:
        mark = "YES" if r[0] in excluded else ""
        big = "yes" if r[1] > MIN_BLOCK_VOXELS else ""
        if prev is not None and prev > MIN_LARGEST_SHARE >= r[4]:
            print(f"{'':-<12}{'':->9}{'':->8}{'':->9}{'':->8}{'':->8}{'':->8}{'':->8}"
                  f"{'':->11}   rule cuts here")
        print(f"{r[0]:<12}{r[1]:>9,}{r[2]:>8}{r[3]:>9,}{r[4]:>8.3f}"
              f"{r[5]:>8}{r[6]:>8}{r[7]:>8}{big:>11}   {mark}")
        prev = r[4]
    if SHOW > 0 and len(ranked) > SHOW:
        print(f"... and {len(ranked) - SHOW} more with at least one missing voxel")

    nNone = sum(1 for r in rows if r[1] == 0)
    total = sum(r[1] for r in rows)
    print(f"\nsubjects with no missing voxel: {nNone} of {len(rows)}")
    print(f"missing mask-voxel slots across the sample: {total:,}")
    print(f"excluded: {len(excluded)}  {', '.join(excluded) if excluded else '(none)'}")
    kept = [r[0] for r in rows if r[0] not in excluded]
    worst = max((r[1] for r in rows if r[0] not in excluded), default=0)
    print(f"retained: {len(kept)}   worst retained gap: {worst:,} voxels")

    os.makedirs(LOG_DIR, exist_ok=True)
    tag = stem(maskPath)
    outCsv = os.path.join(LOG_DIR, f"coverage_from_support_{tag}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["subject", "missing", "pieces", "largest_piece", "share_in_largest",
                      "z_min", "z_median", "z_max"])
        wtr.writerows(rows)
    outTxt = os.path.join(LOG_DIR, f"subjects_excluded_coverage_{tag}.txt")
    with open(outTxt, "w") as fh:
        for s in excluded:
            fh.write(s + "\n")
    print(f"\nwrote {outCsv}")
    print(f"wrote {outTxt}")

    if refPath:
        if not os.path.isfile(refPath):
            sys.exit(f"reference not found: {refPath}")
        ref = {}
        with open(refPath, newline="") as fh:
            for row in csv.DictReader(fh):
                ref[row[REFERENCE_SUBJECT_COLUMN]] = int(row[REFERENCE_VALUE_COLUMN])
        here = {r[0]: r[1] for r in rows}
        shared = sorted(set(ref) & set(here))
        onlyRef = sorted(set(ref) - set(here))
        onlyHere = sorted(set(here) - set(ref))
        diffs = [(abs(ref[s] - here[s]), s, ref[s], here[s]) for s in shared]
        diffs.sort(reverse=True)
        print(f"\nvalidation against {refPath}")
        print(f"  shared {len(shared)}, reference only {len(onlyRef)}, here only {len(onlyHere)}")
        for d, s, a, b in diffs[:5]:
            print(f"  {s:<12} reference {a:>7,}   here {b:>7,}   difference {d:>7,}")
        worstDiff = diffs[0][0] if diffs else 0
        if worstDiff == 0 and not onlyRef and not onlyHere:
            print("  PASS: the stored extents give the same uncovered counts")
        else:
            print(f"  FAIL: largest difference {worstDiff:,}")


if __name__ == "__main__":
    main()
