"""
Compare the two orders of the MNI152NLin2009cAsym-to-MNI152NLin6Asym affine and warp.

IN   priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     priors/xfm_cc/2009cAsym_to_NLin6Asym_{0GenericAffine.mat, 1Warp.nii.gz}
                                          made by register_templates.py cc
     stopsignal/data/derivatives/fmriprep/<sub>/anat/<sub>_space-MNI152NLin2009cAsym_dseg.nii.gz
     subjects_stopsignal_all.txt   one participant ID per line, in this folder

OUT  printed Dice per participant and order
"""

import os
import sys

import numpy as np

import ants

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
SPACE = "space-MNI152NLin2009cAsym"

PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
XFM_DIR = os.path.join(PROJECT_ROOT, "priors", "xfm_cc")
AFFINE = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_0GenericAffine.mat")
WARP = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_1Warp.nii.gz")

ANAT_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "data", "derivatives", "fmriprep")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FILE = os.path.join(SCRIPT_DIR, "subjects_stopsignal_all.txt")

N_TEST = 3
INTERPOLATOR = "genericLabel"
ORDERS = {"[warp, affine]": [WARP, AFFINE],
          "[affine, warp]": [AFFINE, WARP]}
# ====================================================================


def dice(a, b):
    inter = int(np.logical_and(a, b).sum())
    n1, n2 = int(a.sum()), int(b.sum())
    if n1 + n2 == 0:
        return 0.0, 0.0
    return 2 * inter / (n1 + n2), 2 * min(n1, n2) / (n1 + n2)


def main():
    for p in (PRIORS_TEMPLATE, AFFINE, WARP, SUBJECT_FILE):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: not found: {p}")

    subs = [l.strip() for l in open(SUBJECT_FILE) if l.strip()][:N_TEST]
    print(f"testing {len(subs)} subjects: {', '.join(subs)}")
    print(f"interpolator: {INTERPOLATOR}\n")

    ref = ants.image_read(PRIORS_TEMPLATE)
    priors_brain = ref.numpy() > 0
    print(f"priors template: {ref.shape}, {int(priors_brain.sum())} non-zero voxels\n")

    results = {k: [] for k in ORDERS}
    for sub in subs:
        f = os.path.join(ANAT_ROOT, sub, "anat", f"{sub}_{SPACE}_dseg.nii.gz")
        if not os.path.isfile(f):
            print(f"{sub}  SKIPPED - dseg not found")
            continue
        src = ants.threshold_image(ants.image_read(f), 0.5, 1e9)
        print(f"{sub}  native brain mask: {int(src.numpy().sum())} voxels at 1 mm")

        for name, tlist in ORDERS.items():
            out = ants.apply_transforms(fixed=ref, moving=src, transformlist=tlist,
                                        interpolator=INTERPOLATOR)
            m = out.numpy() > 0.5
            d, ceiling = dice(m, priors_brain)
            results[name].append(d)
            print(f"    {name:16s} {int(m.sum()):7d} voxels   Dice {d:.4f}   ceiling {ceiling:.4f}")
        print()

    print("=" * 70)
    for name in ORDERS:
        v = results[name]
        if v:
            print(f"{name:16s} mean Dice {np.mean(v):.4f}")
    best = max((k for k in ORDERS if results[k]), key=lambda k: np.mean(results[k]))
    print(f"\nbetter ordering: {best}")
    print("=" * 70)


if __name__ == "__main__":
    main()
