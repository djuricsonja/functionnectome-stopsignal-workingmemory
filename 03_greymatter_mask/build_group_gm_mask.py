"""
Per-voxel participant counts of grey matter and of functional coverage on the priors grid.

IN   stopsignal/data/derivatives/fmriprep/<sub>/anat/<sub>_space-MNI152NLin2009cAsym_dseg.nii.gz
     stopsignal/data/derivatives/fmriprep/<sub>/func/<sub>_task-stopsignal_acq-seq_space-T1w_desc-preproc_bold.nii.gz
     stopsignal/data/derivatives/fmriprep/<sub>/anat/<sub>_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     priors/xfm_cc/2009cAsym_to_NLin6Asym_{0GenericAffine.mat, 1Warp.nii.gz}
                                          made by register_templates.py cc (02_template_and_route)
     subjects_stopsignal_all.txt   one participant ID per line, in this folder

OUT  stopsignal/masks/group/gm_count.nii.gz
     stopsignal/masks/group/fov_count.nii.gz
     stopsignal/masks/group/agreement_curve.csv
"""

import os
import sys
import time

import numpy as np
import nibabel as nib
import ants

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
SPACE = "space-MNI152NLin2009cAsym"

PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
XFM_DIR = os.path.join(PROJECT_ROOT, "priors", "xfm_cc")
AFFINE = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_0GenericAffine.mat")
WARP = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_1Warp.nii.gz")
TRANSFORMS = [WARP, AFFINE]

ANAT_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "data", "derivatives", "fmriprep")
MASK_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "masks")
OUT_DIR = os.path.join(MASK_ROOT, "group")
TMP_DIR = os.path.join(MASK_ROOT, "tmp_extent")

TASK = "task-stopsignal_acq-seq"
XFM_SUFFIX = "_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FILE = os.path.join(SCRIPT_DIR, "subjects_stopsignal_all.txt")

GM_LABEL = 1
INTERPOLATOR = "genericLabel"
OCCUPANCY = 0.5
AGREEMENT_LEVELS = [1.00, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10]
# ====================================================================


def to_priors(path, ref, low, high, tlist=None):
    img = ants.threshold_image(ants.image_read(path), low, high)
    out = ants.apply_transforms(fixed=ref, moving=img,
                                transformlist=tlist if tlist else TRANSFORMS,
                                interpolator=INTERPOLATOR)
    return out.numpy() > OCCUPANCY


def sampled_extent(bold_path, xfm_path, ref, tmp):
    img = nib.load(bold_path)
    ext = (np.asanyarray(img.dataobj, dtype=np.float32) != 0).any(axis=3)
    nib.save(nib.Nifti1Image(ext.astype(np.uint8), img.affine), tmp)
    out = ants.apply_transforms(fixed=ref, moving=ants.image_read(tmp),
                                transformlist=[WARP, AFFINE, xfm_path],
                                interpolator=INTERPOLATOR)
    os.remove(tmp)
    return out.numpy() > OCCUPANCY


def main():
    for p in (PRIORS_TEMPLATE, AFFINE, WARP, SUBJECT_FILE):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: not found: {p}")

    subs = [l.strip() for l in open(SUBJECT_FILE) if l.strip()]
    print(f"subjects: {len(subs)} (from {os.path.basename(SUBJECT_FILE)})")
    print(f"grey-matter label: {GM_LABEL}   interpolator: {INTERPOLATOR}   "
          f"occupancy: {OCCUPANCY}")
    print(f"transform order: {[os.path.basename(t) for t in TRANSFORMS]}\n")

    ref = ants.image_read(PRIORS_TEMPLATE)
    priors_brain = ref.numpy() > 0
    aff = nib.load(PRIORS_TEMPLATE).affine
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    print(f"priors grid {ref.shape}, brain {int(priors_brain.sum())} voxels")
    print(f"output: {OUT_DIR}\n")

    gm_count = np.zeros(ref.shape, dtype=np.int16)
    fov_count = np.zeros(ref.shape, dtype=np.int16)
    done, skipped = 0, []
    t0 = time.time()

    for i, sub in enumerate(subs, 1):
        dseg = os.path.join(ANAT_ROOT, sub, "anat", f"{sub}_{SPACE}_dseg.nii.gz")
        bold = os.path.join(ANAT_ROOT, sub, "func",
                            f"{sub}_{TASK}_space-T1w_desc-preproc_bold.nii.gz")
        xfm = os.path.join(ANAT_ROOT, sub, "anat", sub + XFM_SUFFIX)
        missing = [p for p in (dseg, bold, xfm) if not os.path.isfile(p)]
        if missing:
            print(f"[{i:3d}/{len(subs)}] {sub}  SKIPPED - missing "
                  f"{os.path.basename(missing[0])}", flush=True)
            skipped.append(sub)
            continue

        gm = to_priors(dseg, ref, GM_LABEL - 0.5, GM_LABEL + 0.5)
        fov = sampled_extent(bold, xfm, ref, os.path.join(TMP_DIR, f"{sub}_ext.nii.gz"))
        gm_count += gm
        fov_count += fov
        done += 1
        el = time.time() - t0
        print(f"[{i:3d}/{len(subs)}] {sub}  GM {int(gm.sum()):6d}  "
              f"coverage {int(fov.sum()):6d}  ({el/i:.1f}s/subject, "
              f"{(len(subs) - i) * el / i / 60:.1f} min left)", flush=True)

    if done == 0:
        sys.exit("ERROR: no subject processed")

    nib.save(nib.Nifti1Image(gm_count, aff), os.path.join(OUT_DIR, "gm_count.nii.gz"))
    nib.save(nib.Nifti1Image(fov_count, aff), os.path.join(OUT_DIR, "fov_count.nii.gz"))

    print("\n" + "=" * 78)
    print(f"processed {done}/{len(subs)}")
    if skipped:
        print(f"SKIPPED: {', '.join(skipped)}")

    print("\ngrey matter in at least one subject: "
          f"{int((gm_count > 0).sum())} voxels")
    print("agreement rule       GM voxels   also covered   inside priors brain")
    rows = []
    for frac in AGREEMENT_LEVELS:
        k = int(np.ceil(frac * done))
        g = gm_count >= k
        both = np.logical_and(g, fov_count >= k)
        inside = np.logical_and(both, priors_brain)
        print(f"{k:4d}/{done} ({frac:5.0%})   {int(g.sum()):9d}   "
              f"{int(both.sum()):12d}   {int(inside.sum()):18d}")
        rows.append((k, frac, int(g.sum()), int(both.sum()), int(inside.sum())))

    with open(os.path.join(OUT_DIR, "agreement_curve.csv"), "w") as fh:
        fh.write("k,fraction,gm_voxels,gm_and_covered,inside_priors_brain\n")
        for k, frac, a, b, c in rows:
            fh.write(f"{k},{frac},{a},{b},{c}\n")

    print(f"\nfor scale, one subject's grey matter on this grid is roughly "
          f"{int(gm_count.sum() / done)} voxels")
    print(f"\nwrote {OUT_DIR}/gm_count.nii.gz")
    print(f"wrote {OUT_DIR}/fov_count.nii.gz")
    print(f"wrote {OUT_DIR}/agreement_curve.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()
