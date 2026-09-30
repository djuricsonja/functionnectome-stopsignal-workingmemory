"""
Write the group anatomical label map from the per-structure participant counts.

IN   priors | arm1                    target grid
     level                            optional share of participants, default 0.5
     nomerge                          optional, keeps every cortical parcel separate
     stopsignal/masks/aseg/<label>_<name>_count.nii.gz        priors, made by build_group_aseg_counts.py priors
     stopsignal/masks/aseg_arm1/<label>_<name>_count.nii.gz   arm1, made by build_group_aseg_counts.py arm1
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     stopsignal/group_all/SuccStop_vs_Go/mask.nii   made by group_level_stop_signal.m arm1 (01_greymatter_glm)

OUT  aseg_labels_<level>pc.nii.gz, aseg_labels_<level>pc.csv, in the count folder
     (aseg_labels_<level>pc_nomerge.* with nomerge)
"""

import csv
import os
import re
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
MASK_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "masks")

TARGETS = {
    "priors": {
        "reference": os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz"),
        "out": os.path.join(MASK_ROOT, "aseg"),
    },
    "arm1": {
        "reference": os.path.join(PROJECT_ROOT, "stopsignal", "group_all",
                                  "SuccStop_vs_Go", "mask.nii"),
        "out": os.path.join(MASK_ROOT, "aseg_arm1"),
    },
}

COUNT_PATTERN = re.compile(r"^(\d{4})_(.+)_count\.nii\.gz$")
OUT_STEM = "aseg_labels_{:d}pc"
MERGE_STEM = "aseg_labels_{:d}pc_nomerge"

# (first, last) label values, the value and name they become
MERGE_RANGES = [((1000, 1999), 3, "Left-Cerebral-Cortex"),
                ((2000, 2999), 42, "Right-Cerebral-Cortex")]

LEVEL = 0.50
# ====================================================================


def merge(counts, names):
    for (lo, hi), value, name in MERGE_RANGES:
        members = [v for v in counts if lo <= v <= hi]
        if not members:
            continue
        total = np.zeros_like(counts[members[0]])
        for v in members:
            total += counts.pop(v)
            names.pop(v, None)
        counts[value] = total
        names[value] = name
        print(f"merged {len(members):>4} labels {lo}-{hi} into {value} {name}")


def main():
    args = sys.argv[1:]
    doMerge = "nomerge" not in args
    args = [a for a in args if a != "nomerge"]
    if not args or args[0] not in TARGETS:
        sys.exit("give the target: " + ", ".join(sorted(TARGETS)))
    target = args[0]
    ASEG_DIR = TARGETS[target]["out"]
    PRIORS_TEMPLATE = TARGETS[target]["reference"]
    level = float(args[1]) if len(args) > 1 else LEVEL
    if not 0 < level <= 1:
        sys.exit(f"level must be above 0 and at most 1, got {level}")
    if not os.path.isdir(ASEG_DIR):
        sys.exit(f"ERROR: not found: {ASEG_DIR}")
    files = sorted(f for f in os.listdir(ASEG_DIR) if COUNT_PATTERN.match(f))
    if not files:
        sys.exit(f"ERROR: no count maps in {ASEG_DIR}")

    tpl = nib.load(PRIORS_TEMPLATE)
    ref = np.asanyarray(tpl.dataobj)

    print(f"target:      {target}")
    print(f"reference:   {PRIORS_TEMPLATE}")
    print(f"count maps:  {len(files)} in {ASEG_DIR}")
    print(f"level:       {level:.0%}")

    counts = {}
    names = {}
    nSubjects = 0
    for f in files:
        m = COUNT_PATTERN.match(f)
        value, name = int(m.group(1)), m.group(2)
        img = nib.load(os.path.join(ASEG_DIR, f))
        if img.shape[:3] != tpl.shape[:3] or not np.allclose(img.affine, tpl.affine):
            sys.exit(f"ERROR: {f} is not on the target grid")
        arr = np.asanyarray(img.dataobj)
        counts[value] = arr
        names[value] = name
        nSubjects = max(nSubjects, int(arr.max()))

    if doMerge:
        merge(counts, names)
        print()

    k = int(np.ceil(level * nSubjects))
    exclusive = 2 * k > nSubjects        # two structures cannot both reach k
    print(f"subjects:    {nSubjects} (read from the count maps)")
    print(f"threshold:   at least {k} of {nSubjects}, "
          f"{'one structure per voxel is guaranteed' if exclusive else 'overlap is possible'}\n")

    labels = np.zeros(tpl.shape[:3], dtype=np.int16)
    hits = np.zeros(tpl.shape[:3], dtype=np.int16)
    for value in sorted(counts):
        sel = counts[value] >= k
        hits += sel
        labels[sel] = value

    clash = int((hits > 1).sum())
    if clash and exclusive:
        sys.exit(f"ERROR: {clash} voxels reach the threshold for more than one structure "
                 f"at {level:.0%}; the counts cannot all come from one label image per subject")
    if clash:
        print(f"note: {clash} voxels reach the threshold for more than one structure at "
              f"{level:.0%}; the last structure by label number wins\n")

    rows = []
    print(f"{'label':>6}  {'name':<34}{'voxels':>10}{'ml':>9}")
    voxml = float(np.prod(tpl.header.get_zooms()[:3])) / 1000.0
    for value in sorted(counts):
        n = int((labels == value).sum())
        rows.append([value, names[value], n, round(n * voxml, 2)])
        if n:
            print(f"{value:>6}  {names[value]:<34}{n:>10,}{n * voxml:>9.1f}")

    assigned = int((labels > 0).sum())
    inBrain = int(((labels > 0) & (ref > 0)).sum())
    print(f"\nassigned {assigned:,} voxels, {inBrain:,} of them inside the priors brain "
          f"({ref.astype(bool).sum():,} voxels)")

    for (lo, hi), value, name in MERGE_RANGES:
        if not doMerge or value not in counts:
            continue
        side = name.split("-")[0]
        ijk = np.argwhere(labels == value)
        if ijk.size == 0:
            continue
        x = nib.affines.apply_affine(tpl.affine, ijk)[:, 0].mean()
        if (x > 0) != (side == "Right"):
            sys.exit(f"ERROR: {name} has its centre of mass at x = {x:.1f} mm, "
                     f"which is not the {side} hemisphere; the merged ranges are wrong")
        print(f"{name} centre of mass x = {x:+.1f} mm, in the {side} hemisphere")

    stem = (OUT_STEM if doMerge else MERGE_STEM).format(int(round(level * 100)))
    outNii = os.path.join(ASEG_DIR, stem + ".nii.gz")
    outCsv = os.path.join(ASEG_DIR, stem + ".csv")
    nib.save(nib.Nifti1Image(labels, tpl.affine), outNii)
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["label", "name", "voxels", "ml", "n_subjects", "threshold"])
        for r in rows:
            wtr.writerow(r + [nSubjects, k])

    print(f"\nwrote {outNii}")
    print(f"wrote {outCsv}")


if __name__ == "__main__":
    main()
