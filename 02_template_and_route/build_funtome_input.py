"""
Carry one participant's T1w-space BOLD onto the priors grid.

IN   <sub>                     argument 1
     <interpolator>            argument 2, ANTs interpolator name
     [out dir]                 argument 3, optional
     stopsignal/data/derivatives/fmriprep/<sub>/func/<sub>_task-stopsignal_acq-seq_space-T1w_desc-preproc_bold.nii.gz
     stopsignal/data/derivatives/fmriprep/<sub>/anat/<sub>_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5
     priors/xfm_cc/2009cAsym_to_NLin6Asym_{1Warp.nii.gz, 0GenericAffine.mat}
                                          made by register_templates.py cc
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  <out dir>/<sub>.nii.gz            BOLD on the priors grid
     <out dir>/<sub>_support.nii.gz    scanned field of view on the priors grid
     printed checks
"""

import os
import sys
import time

import numpy as np
import nibabel as nib

import ants

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
FMRIPREP = os.path.join(PROJECT_ROOT, "stopsignal", "data", "derivatives", "fmriprep")
PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
XFM_DIR = os.path.join(PROJECT_ROOT, "priors", "xfm_cc")
AFFINE = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_0GenericAffine.mat")
WARP = os.path.join(XFM_DIR, "2009cAsym_to_NLin6Asym_1Warp.nii.gz")

DEFAULT_OUT = os.path.join(PROJECT_ROOT, "stopsignal", "functionnectome", "input")
TASK = "task-stopsignal_acq-seq"
SUPPORT_INTERPOLATOR = "genericLabel"
# ====================================================================


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: build_funtome_input.py <subject> <interpolator> [out dir]")
    sub, interp = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_OUT

    bold_f = os.path.join(FMRIPREP, sub, "func",
                          f"{sub}_{TASK}_space-T1w_desc-preproc_bold.nii.gz")
    xfm_f = os.path.join(FMRIPREP, sub, "anat",
                         f"{sub}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5")
    for p in (bold_f, xfm_f, PRIORS_TEMPLATE, AFFINE, WARP):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: not found: {p}")
    os.makedirs(out_dir, exist_ok=True)

    tlist = [WARP, AFFINE, xfm_f]
    ref = ants.image_read(PRIORS_TEMPLATE)
    print(f"{sub}  interpolator {interp}")
    print(f"  reference {ref.shape}, spacing {tuple(round(s, 3) for s in ref.spacing)}")

    src_img = nib.load(bold_f)
    tr = float(src_img.header["pixdim"][4])
    print(f"  source {src_img.shape}, TR {tr}", flush=True)

    print("  building the source support...", flush=True)
    src = np.asanyarray(src_img.dataobj, dtype=np.float32)
    src_support = src.any(axis=3)
    src_min, src_max = float(src.min()), float(src.max())
    print(f"  source sampled in {int(src_support.sum())} of {src_support.size} voxels; "
          f"values {src_min:.4f} to {src_max:.4f}")
    sup_src_f = os.path.join(out_dir, f"{sub}_support_T1w.nii.gz")
    nib.save(nib.Nifti1Image(src_support.astype(np.uint8), src_img.affine), sup_src_f)
    src = None

    print("  resampling BOLD...", flush=True)
    t0 = time.time()
    out = ants.apply_transforms(fixed=ref, moving=ants.image_read(bold_f),
                                transformlist=tlist, interpolator=interp, imagetype=3)
    out_f = os.path.join(out_dir, f"{sub}.nii.gz")
    ants.image_write(out, out_f)
    out = None
    print(f"  wrote {out_f} in {time.time() - t0:.0f} s", flush=True)

    print(f"  resampling the support with {SUPPORT_INTERPOLATOR}...", flush=True)
    msk = ants.apply_transforms(fixed=ref, moving=ants.image_read(sup_src_f),
                                transformlist=tlist, interpolator=SUPPORT_INTERPOLATOR)
    sup_f = os.path.join(out_dir, f"{sub}_support.nii.gz")
    ants.image_write(msk, sup_f)
    os.remove(sup_src_f)

    img = nib.load(out_f)
    hdr = img.header.copy()
    hdr["pixdim"][4] = tr
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, hdr), out_f)

    print("  checking the result...", flush=True)
    img = nib.load(out_f)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    support = np.asanyarray(nib.load(sup_f).dataobj) > 0.5
    tpl = np.asanyarray(nib.load(PRIORS_TEMPLATE).dataobj) > 0

    if not np.allclose(img.affine, nib.load(PRIORS_TEMPLATE).affine):
        print("  WARNING: output affine differs from the priors template")
    print(f"  shape {data.shape}, dtype {data.dtype}, "
          f"TR {float(img.header['pixdim'][4])}")
    print(f"  support {int(support.sum())} voxels; "
          f"priors template {int(tpl.sum())} voxels; "
          f"support inside template {int(np.logical_and(support, tpl).sum())}")

    nz = data.any(axis=3)
    outside = np.logical_and(nz, ~support)
    print(f"  RESULT nonzero voxels                {int(nz.sum())}")
    print(f"  RESULT nonzero outside the support   {int(outside.sum())}")
    if outside.any():
        print(f"  RESULT largest value out there       "
              f"{np.abs(data[outside]).max():.4f}")
    print(f"  RESULT nonzero outside priors brain   "
          f"{int(np.logical_and(nz, ~tpl).sum())}")

    inside = data[support]
    print(f"  RESULT inside support: min {inside.min():.4f}, max {inside.max():.4f}, "
          f"mean {inside.mean():.4f}")
    print(f"  RESULT below the source minimum      "
          f"{int((inside < src_min).sum())} of {inside.size} samples")
    print(f"  RESULT above the source maximum      "
          f"{int((inside > src_max).sum())} of {inside.size} samples")
    print(f"  RESULT negative values in support    {int((inside < 0).sum())}")
    print(f"  RESULT NaN {int(np.isnan(data).sum())}, Inf {int(np.isinf(data).sum())}")


if __name__ == "__main__":
    main()
