"""
Write the group grey-matter mask from the anatomical parcellation counts.

IN   gm_level                         optional share of participants, default 0.5
     fov_level                        optional share of participants, default 0.95
     stopsignal/masks/aseg/<label>_<name>_count.nii.gz   made by build_group_aseg_counts.py priors
     FreeSurferSubcorticalLabelTableLut.txt   from HCP Pipelines v6.0.0, global/config/, in this folder
     stopsignal/masks/group/fov_count.nii.gz   made by build_group_gm_mask.py
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     stopsignal/masks/group/group_gm_mask.nii.gz   made by write_group_gm_mask.py, for comparison only

OUT  stopsignal/masks/group_aseg/group_gm_mask.nii.gz
"""

import os
import re
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
MASK_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "masks")

COUNT_DIR = os.path.join(MASK_ROOT, "aseg")
FOV_COUNT = os.path.join(MASK_ROOT, "group", "fov_count.nii.gz")
OLD_MASK = os.path.join(MASK_ROOT, "group", "group_gm_mask.nii.gz")
OUT_DIR = os.path.join(MASK_ROOT, "group_aseg")
OUT_MASK = os.path.join(OUT_DIR, "group_gm_mask.nii.gz")

PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBCORTICAL_LUT = os.path.join(SCRIPT_DIR, "FreeSurferSubcorticalLabelTableLut.txt")

COUNT_PATTERN = re.compile(r"^(\d{4})_(.+)_count\.nii\.gz$")
CORTEX_RANGES = ((1000, 1999), (2000, 2999))

GM_LEVEL = 0.50
FOV_LEVEL = 0.95
# ====================================================================


def read_subcortical_labels(path):
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
    gmLevel = float(sys.argv[1]) if len(sys.argv) > 1 else GM_LEVEL
    fovLevel = float(sys.argv[2]) if len(sys.argv) > 2 else FOV_LEVEL
    for name, lv in (("gm_level", gmLevel), ("fov_level", fovLevel)):
        if not 0 < lv <= 1:
            sys.exit(f"{name} must be above 0 and at most 1, got {lv}")

    for p in (COUNT_DIR, SUBCORTICAL_LUT, FOV_COUNT, PRIORS_TEMPLATE):
        if not os.path.exists(p):
            sys.exit(f"ERROR: not found: {p}")

    tpl = nib.load(PRIORS_TEMPLATE)
    brain = np.asanyarray(tpl.dataobj) > 0

    files = {}
    for f in sorted(os.listdir(COUNT_DIR)):
        m = COUNT_PATTERN.match(f)
        if m:
            files[int(m.group(1))] = f

    subcortical = read_subcortical_labels(SUBCORTICAL_LUT)
    greyLabels = {v: subcortical[v] for v in subcortical}
    nCortex = 0
    for v in files:
        if any(lo <= v <= hi for lo, hi in CORTEX_RANGES):
            greyLabels[v] = f"cortical_parcel_{v}"
            nCortex += 1

    print(f"count maps:   {COUNT_DIR}")
    print(f"colour table: {SUBCORTICAL_LUT}")
    print(f"grey labels:  {len(greyLabels)} = {len(subcortical)} subcortical + "
          f"{nCortex} cortical parcels")

    total = None
    missing = []
    for v in sorted(greyLabels):
        if v not in files:
            missing.append(f"{v}:{greyLabels[v]}")
            continue
        img = nib.load(os.path.join(COUNT_DIR, files[v]))
        if img.shape[:3] != tpl.shape[:3] or not np.allclose(img.affine, tpl.affine):
            sys.exit(f"ERROR: {files[v]} is not on the priors grid")
        total = np.asanyarray(img.dataobj).astype(np.int32) if total is None \
            else total + np.asanyarray(img.dataobj).astype(np.int32)
    if total is None:
        sys.exit("none of the grey labels has a count map")
    if missing:
        sys.exit(f"no count map for {len(missing)} grey labels: {missing[:10]}")

    fovImg = nib.load(FOV_COUNT)
    if fovImg.shape[:3] != tpl.shape[:3] or not np.allclose(fovImg.affine, tpl.affine):
        sys.exit("ERROR: fov_count is not on the priors grid")
    fov = np.asanyarray(fovImg.dataobj).astype(np.int32)

    nSubjects = int(max(total.max(), fov.max()))
    kGm = int(np.ceil(gmLevel * nSubjects))
    kFov = int(np.ceil(fovLevel * nSubjects))
    print(f"participants: {nSubjects} (read from the count maps)")
    print(f"grey rule:    at least {kGm} of {nSubjects} ({gmLevel:.0%})")
    print(f"coverage:     at least {kFov} of {nSubjects} ({fovLevel:.0%})\n")

    a = total >= kGm
    b = np.logical_and(a, fov >= kFov)
    c = np.logical_and(b, brain)

    print(f"priors brain: {int(brain.sum()):,} voxels")
    print(f"grey in >= {kGm}/{nSubjects}:            {int(a.sum()):>8,}")
    print(f"  and covered in >= {kFov}/{nSubjects}:  {int(b.sum()):>8,}")
    print(f"  and inside the priors brain:       {int(c.sum()):>8,}")

    prof = c.sum(axis=(0, 1))
    z = np.nonzero(prof)[0]
    if z.size:
        zmm = nib.affines.apply_affine(tpl.affine, np.array([[0, 0, z.min()],
                                                             [0, 0, z.max()]]))[:, 2]
        print(f"  z extent: {zmm[0]:.0f} to {zmm[1]:.0f} mm")

    if os.path.exists(OLD_MASK):
        old = np.asanyarray(nib.load(OLD_MASK).dataobj) > 0
        print(f"\nexisting mask {OLD_MASK}")
        print(f"  voxels:            {int(old.sum()):>8,}")
        print(f"  kept by both:      {int(np.logical_and(old, c).sum()):>8,}")
        print(f"  added by the new:  {int(np.logical_and(~old, c).sum()):>8,}")
        print(f"  dropped from old:  {int(np.logical_and(old, ~c).sum()):>8,}")

    os.makedirs(OUT_DIR, exist_ok=True)
    nib.save(nib.Nifti1Image(c.astype(np.uint8), tpl.affine), OUT_MASK)
    print(f"\nwrote {OUT_MASK}")


if __name__ == "__main__":
    main()
