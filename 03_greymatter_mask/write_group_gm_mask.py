"""
Write the group grey-matter mask from the tissue-segmentation counts.

IN   stopsignal/masks/group/gm_count.nii.gz    made by build_group_gm_mask.py
     stopsignal/masks/group/fov_count.nii.gz   made by build_group_gm_mask.py
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  stopsignal/masks/group/group_gm_mask.nii.gz
"""

import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
GROUP_DIR = os.path.join(PROJECT_ROOT, "stopsignal", "masks", "group")
PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")

GM_COUNT = os.path.join(GROUP_DIR, "gm_count.nii.gz")
FOV_COUNT = os.path.join(GROUP_DIR, "fov_count.nii.gz")
OUT_MASK = os.path.join(GROUP_DIR, "group_gm_mask.nii.gz")

GM_LEVEL = 0.50
FOV_LEVEL = 0.95
# ====================================================================


def main():
    for p in (GM_COUNT, FOV_COUNT, PRIORS_TEMPLATE):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: not found: {p}")

    gm_img, fov_img = nib.load(GM_COUNT), nib.load(FOV_COUNT)
    tpl_img = nib.load(PRIORS_TEMPLATE)
    for name, img in (("fov_count", fov_img), ("priors template", tpl_img)):
        if img.shape[:3] != gm_img.shape[:3] or not np.allclose(img.affine, gm_img.affine):
            sys.exit(f"ERROR: {name} is not on the same grid as gm_count")

    gm = np.asanyarray(gm_img.dataobj)
    fov = np.asanyarray(fov_img.dataobj)
    brain = np.asanyarray(tpl_img.dataobj) > 0

    n_subjects = int(max(gm.max(), fov.max()))
    k_gm = int(np.ceil(GM_LEVEL * n_subjects))
    k_fov = int(np.ceil(FOV_LEVEL * n_subjects))
    print(f"subject count read from the count maps: {n_subjects}")

    a = gm >= k_gm
    b = np.logical_and(a, fov >= k_fov)
    c = np.logical_and(b, brain)

    print(f"grid {gm_img.shape}, priors brain {int(brain.sum())} voxels")
    print(f"grey matter in >= {k_gm}/{n_subjects} ({GM_LEVEL:.0%}):        {int(a.sum()):7d}")
    print(f"  and covered in >= {k_fov}/{n_subjects} ({FOV_LEVEL:.0%}):    {int(b.sum()):7d}")
    print(f"  and inside the priors brain:              {int(c.sum()):7d}")

    prof = c.sum(axis=(0, 1))
    z = np.nonzero(prof)[0]
    aff = gm_img.affine
    print(f"axial extent: z = {aff[2, 3] + aff[2, 2] * z[0]:.0f} to "
          f"{aff[2, 3] + aff[2, 2] * z[-1]:.0f} mm")

    nib.save(nib.Nifti1Image(c.astype(np.uint8), aff), OUT_MASK)
    check = nib.load(OUT_MASK)
    print(f"\nwrote {OUT_MASK}")
    print(f"  shape {check.shape}, dtype {check.get_data_dtype()}, "
          f"{int(np.asanyarray(check.dataobj).sum())} voxels")
    print(f"  affine matches the priors template: "
          f"{np.allclose(check.affine, tpl_img.affine)}")


if __name__ == "__main__":
    main()
