"""
Grey-matter mask size at every agreement level, from the anatomical parcellation counts.

IN   priors                           target grid
     stopsignal/masks/aseg/<label>_<name>_count.nii.gz   made by build_group_aseg_counts.py priors
     FreeSurferSubcorticalLabelTableLut.txt   from HCP Pipelines v6.0.0, global/config/, in this folder
     stopsignal/masks/group/fov_count.nii.gz   made by build_group_gm_mask.py
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  logs/aseg_agreement_curve_priors.csv
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
    "priors": os.path.join(MASK_ROOT, "aseg"),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBCORTICAL_LUT = os.path.join(SCRIPT_DIR, "FreeSurferSubcorticalLabelTableLut.txt")

PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
FOV_COUNT = os.path.join(MASK_ROOT, "group", "fov_count.nii.gz")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

COUNT_PATTERN = re.compile(r"^(\d{4})_(.+)_count\.nii\.gz$")
CORTEX_RANGES = ((1000, 1999), (2000, 2999))   # the parcellation's own cortical parcels
FOV_LEVEL = 0.95
STEP = 5
# ====================================================================


def read_subcortical_labels(path):
    """Label values from the colour table: a name line, then a line starting with the value."""
    values = {}
    name = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0].lstrip("-").isdigit() and name is not None:
                values[int(parts[0])] = name
                name = None
            else:
                name = line
    return values


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        sys.exit("give the target: " + ", ".join(sorted(TARGETS)))
    target = sys.argv[1]
    countDir = TARGETS[target]

    for p in (countDir, SUBCORTICAL_LUT, PRIORS_TEMPLATE):
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found: {p}")

    subcortical = read_subcortical_labels(SUBCORTICAL_LUT)

    print(f"target:      {target}")
    print(f"count maps:  {countDir}")
    print(f"colour table: {SUBCORTICAL_LUT}")

    files = {}
    for f in sorted(os.listdir(countDir)):
        m = COUNT_PATTERN.match(f)
        if m:
            files[int(m.group(1))] = f

    greyLabels = dict(subcortical)
    nCortex = 0
    for v in files:
        if any(lo <= v <= hi for lo, hi in CORTEX_RANGES):
            greyLabels[v] = f"cortical_parcel_{v}"
            nCortex += 1
    print(f"grey labels: {len(greyLabels)} = {len(subcortical)} subcortical from the table "
          f"+ {nCortex} cortical parcels in {CORTEX_RANGES}")
    print("   subcortical: "
          + ", ".join(f"{v}:{subcortical[v]}" for v in sorted(subcortical)) + "\n")

    tpl = nib.load(PRIORS_TEMPLATE)
    brain = np.asanyarray(tpl.dataobj) > 0

    total = None
    used, missing = [], []
    for v in sorted(greyLabels):
        if v not in files:
            missing.append(f"{v}:{greyLabels[v]}")
            continue
        img = nib.load(os.path.join(countDir, files[v]))
        if img.shape[:3] != tpl.shape[:3] or not np.allclose(img.affine, tpl.affine):
            sys.exit(f"ERROR: {files[v]} is not on the same grid as the template")
        arr = np.asanyarray(img.dataobj).astype(np.int32)
        total = arr.copy() if total is None else total + arr
        used.append(f"{v}:{greyLabels[v]}")

    if total is None:
        sys.exit("none of the grey labels has a count map")
    print(f"count maps found for {len(used)} of {len(greyLabels)} grey labels")
    if missing:
        print(f"MISSING: {', '.join(missing)}")

    nSubjects = int(total.max())
    kFov = int(np.ceil(FOV_LEVEL * nSubjects))
    if os.path.exists(FOV_COUNT):
        fov = np.asanyarray(nib.load(FOV_COUNT).dataobj) >= kFov
        haveFov = True
    else:
        fov = np.ones_like(brain)
        haveFov = False
        print(f"NOTE: {FOV_COUNT} not found; the second column is anatomy and template only")

    print(f"\nsubjects: {nSubjects}   coverage rule: at least {kFov} of {nSubjects} "
          f"({FOV_LEVEL:.0%}){'' if haveFov else ' - NOT APPLIED'}")
    print(f"priors brain: {int(brain.sum()):,} voxels\n")

    rows = []
    prev = None
    print(f"{'k':>5}{'level':>8}{'anatomy':>12}{'+coverage+brain':>18}{'change':>10}")
    for k in range(1, nSubjects + 1):
        anat = int((total >= k).sum())
        final = int((np.logical_and(total >= k, fov) & brain).sum())
        rows.append([k, round(100 * k / nSubjects, 2), anat, final])
        if k % STEP == 0 or k in (1, nSubjects):
            change = "" if prev is None else f"{final - prev:+,}"
            print(f"{k:>5}{100 * k / nSubjects:>7.1f}%{anat:>12,}{final:>18,}{change:>10}")
            prev = final

    os.makedirs(LOG_DIR, exist_ok=True)
    outCsv = os.path.join(LOG_DIR, f"aseg_agreement_curve_{target}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["k", "percent", "voxels_anatomy", "voxels_with_coverage_and_brain"])
        wtr.writerows(rows)
    print(f"\nwrote {outCsv}")


if __name__ == "__main__":
    main()
