"""
Per-voxel participant counts of each anatomical structure, on the priors grid or the arm 1 grid.

IN   priors | arm1                    target grid
     n_subjects                       optional, first n participants only
     stopsignal/data_aseg/<sub>_lausanne250+aseg_in_MNI_2mm.nii.gz
                                      from the dataset's fMRIPrep derivatives, <sub>/anat/
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     priors/xfm_cc/2009cAsym_to_NLin6Asym_{0GenericAffine.mat, 1Warp.nii.gz}
                                          made by register_templates.py cc (02_template_and_route)
     stopsignal/group_all/SuccStop_vs_Go/mask.nii   made by group_level_stop_signal.m arm1 (01_greymatter_glm)
     $FREESURFER_HOME/FreeSurferColorLUT.txt
     subjects_stopsignal_all.txt   one participant ID per line, in this folder

OUT  stopsignal/masks/aseg/<label>_<name>_count.nii.gz        priors
     stopsignal/masks/aseg_arm1/<label>_<name>_count.nii.gz   arm1
     aseg_agreement_curve.csv, in the same folder
"""

import csv
import os
import sys
import time

import numpy as np
import nibabel as nib
import ants

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]

ASEG_DIR = os.path.join(PROJECT_ROOT, "stopsignal", "data_aseg")
ASEG_SUFFIX = "_lausanne250+aseg_in_MNI_2mm.nii.gz"

XFM_DIR = os.path.join(PROJECT_ROOT, "priors", "xfm_cc")
AFFINE = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_0GenericAffine.mat")
WARP = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_1Warp.nii.gz")

MASK_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "masks")

TARGETS = {
    "priors": {
        "reference": os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz"),
        "transforms": [WARP, AFFINE],
        "out": os.path.join(MASK_ROOT, "aseg"),
    },
    "arm1": {
        "reference": os.path.join(PROJECT_ROOT, "stopsignal", "group_all",
                                  "SuccStop_vs_Go", "mask.nii"),
        "transforms": [],
        "out": os.path.join(MASK_ROOT, "aseg_arm1"),
    },
}

LUT = os.path.join(os.environ["FREESURFER_HOME"], "FreeSurferColorLUT.txt")
CURVE_FILE = "aseg_agreement_curve.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FILE = os.path.join(SCRIPT_DIR, "subjects_stopsignal_all.txt")

INTERPOLATOR = "genericLabel"
LUT_MAX = 1000                        # below this the input's numbering is FreeSurfer's
AGREEMENT_LEVELS = [1.00, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10]
REPORT_EVERY = 20
# ====================================================================


def subjects_at(level, nSubjects):
    return max(1, int(np.ceil(level * nSubjects)))


def read_lut(path):
    names = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            names[int(parts[0])] = parts[1]
    return names


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("give the target: " + ", ".join(sorted(TARGETS)))
    target = sys.argv[1]
    if target not in TARGETS:
        sys.exit(f"unknown target {target!r}; give one of " + ", ".join(sorted(TARGETS)))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    spec = TARGETS[target]
    reference, transforms, outDir = spec["reference"], spec["transforms"], spec["out"]

    for p in [reference, SUBJECT_FILE, LUT, ASEG_DIR] + list(transforms):
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found: {p}")

    subs = [l.strip() for l in open(SUBJECT_FILE) if l.strip()]
    if limit:
        subs = subs[:limit]

    lutNames = read_lut(LUT)
    ref = ants.image_read(reference)
    aff = nib.load(reference).affine
    shape = ref.numpy().shape

    print(f"target:       {target}")
    print(f"reference:    {reference}")
    print(f"subjects:     {len(subs)} (from {os.path.basename(SUBJECT_FILE)})")
    print(f"input:        {ASEG_DIR}/<sub>{ASEG_SUFFIX}")
    print(f"colour table: {LUT}  ({len(lutNames)} names)")
    print(f"interpolator: {INTERPOLATOR}   labels counted: every label above zero")
    print(f"colour table used for labels below {LUT_MAX}; above it the input's own numbering")
    print(f"transforms:   {[os.path.basename(t) for t in transforms] or 'none, resampling only'}")
    print(f"target grid:  {shape}, voxel "
          f"{tuple(round(float(v), 2) for v in nib.load(reference).header.get_zooms()[:3])} mm")
    print(f"output:       {outDir}\n")

    counts = {}
    done, skipped = 0, []
    t0 = time.time()

    for i, sub in enumerate(subs, 1):
        src = os.path.join(ASEG_DIR, sub + ASEG_SUFFIX)
        if not os.path.isfile(src):
            print(f"{sub}  MISSING {src}")
            skipped.append(sub)
            continue
        moved = ants.apply_transforms(fixed=ref, moving=ants.image_read(src),
                                      transformlist=transforms,
                                      interpolator=INTERPOLATOR)
        lab = np.rint(moved.numpy()).astype(np.int32)
        present = [int(v) for v in np.unique(lab) if v > 0]
        for v in present:
            if v not in counts:
                counts[v] = np.zeros(shape, dtype=np.int16)
            counts[v][lab == v] += 1
        done += 1
        if i % REPORT_EVERY == 0 or i == len(subs):
            print(f"{sub}  {i}/{len(subs)}  labels {len(present)}  "
                  f"structures so far {len(counts)}  {time.time() - t0:.0f}s")

    if not counts:
        sys.exit("no labels counted; nothing to write")

    os.makedirs(outDir, exist_ok=True)
    rows = []
    print("\nagreement levels need " +
          ", ".join(f"{int(100 * L)}%={subjects_at(L, done)}" for L in AGREEMENT_LEVELS)
          + f" of {done} subjects")
    print(f"\n{'label':>6}  {'name':<34}{'max count':>10}" +
          "".join(f"{int(100 * L):>8}%" for L in AGREEMENT_LEVELS))
    for v in sorted(counts):
        name = lutNames.get(v, f"label-{v}") if v < LUT_MAX else f"label-{v}"
        arr = counts[v]
        nib.save(nib.Nifti1Image(arr, aff),
                 os.path.join(outDir, f"{v:04d}_{name}_count.nii.gz"))
        atLevel = [int((arr >= subjects_at(L, done)).sum()) for L in AGREEMENT_LEVELS]
        rows.append([v, name, int(arr.max())] + atLevel)
        print(f"{v:>6}  {name:<34}{int(arr.max()):>10}" +
              "".join(f"{n:>9,}" for n in atLevel))

    with open(os.path.join(outDir, CURVE_FILE), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["label", "name", "max_count", "n_subjects"] +
                     [f"voxels_at_{int(100 * L)}pc" for L in AGREEMENT_LEVELS])
        for r in rows:
            wtr.writerow(r[:3] + [done] + r[3:])

    print(f"\nsubjects counted {done}, skipped {len(skipped)}"
          + (": " + ", ".join(skipped) if skipped else ""))
    print(f"structures written {len(counts)} to {outDir}")
    print(f"agreement curve: {os.path.join(outDir, CURVE_FILE)}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
