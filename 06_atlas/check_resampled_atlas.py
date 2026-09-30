"""
Check a 2 mm resampled atlas against the priors grid and against its 1 mm original.

IN   resampled                        atlases/xtract_2.0.0/xtract-tract-atlases-prob-2mm_flirt_nn.nii.gz, made by resample_atlas_flirt.sh
     original                         atlases/xtract_2.0.0/xtract-tract-atlases-prob-1mm.nii.gz, FSL's XTRACT atlas release 2.0.0
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI

OUT  printed checks and PASS or FAIL
"""

import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
CENTRE_TOLERANCE = 1e-4     # in original voxels
# ====================================================================


def main():
    if len(sys.argv) != 3:
        sys.exit("give the resampled atlas and its original")
    outPath, origPath = sys.argv[1:3]
    for p in (outPath, origPath, PRIORS_TEMPLATE):
        if not os.path.isfile(p):
            sys.exit(f"missing: {p}")
    out = nib.load(outPath)
    orig = nib.load(origPath)
    tpl = nib.load(PRIORS_TEMPLATE)

    nOut = out.shape[3] if out.ndim == 4 else 1
    nOrig = orig.shape[3] if orig.ndim == 4 else 1
    print(f"resampled: {outPath}  shape {out.shape}  dtype {out.get_data_dtype()}")
    print(f"original:  {origPath}  shape {orig.shape}")
    print(f"template:  {PRIORS_TEMPLATE}  shape {tpl.shape}")
    print(f"resampled affine:\n{np.round(out.affine, 4)}")
    print(f"template affine:\n{np.round(tpl.affine, 4)}")
    print(f"sform code {int(out.header['sform_code'])}, qform code {int(out.header['qform_code'])}")

    shapeOk = out.shape[:3] == tpl.shape[:3]
    affineOk = np.array_equal(out.affine, tpl.affine)
    countOk = nOut == nOrig
    print(f"\nshape equals template:   {shapeOk}")
    print(f"affine equals template:  {affineOk}")
    print(f"volume count {nOut} vs original {nOrig}: {countOk}")

    I, J, K = np.meshgrid(*[np.arange(n) for n in out.shape[:3]], indexing="ij")
    centres = np.stack([I.ravel(), J.ravel(), K.ravel(), np.ones(I.size)])
    toOrig = np.linalg.inv(orig.affine) @ out.affine @ centres
    frac = np.abs(toOrig[:3] - np.round(toOrig[:3]))
    print(f"\nlargest distance of an output centre from an original centre: {frac.max():.6f} voxels")
    if frac.max() > CENTRE_TOLERANCE:
        print("centres do not land on original voxel centres; value comparison not made")
        print("VERDICT: FAIL")
        return
    src = np.round(toOrig[:3]).astype(int)
    inside = np.all((src >= 0) & (src < np.array(orig.shape[:3])[:, None]), axis=0)
    print(f"output centres inside the original grid: {int(inside.sum()):,} of {inside.size:,}")

    totalBad = 0
    for v in range(nOut):
        o = np.asanyarray(out.dataobj[..., v] if out.ndim == 4 else out.dataobj, dtype=np.float64).ravel()
        g = np.asanyarray(orig.dataobj[..., v] if orig.ndim == 4 else orig.dataobj, dtype=np.float64)
        expect = np.zeros(o.size)
        expect[inside] = g[src[0, inside], src[1, inside], src[2, inside]]
        bad = int((o != expect).sum())
        totalBad += bad
        if bad:
            print(f"  volume {v}: {bad:,} voxels differ, largest difference {np.abs(o - expect).max():.6g}")
    print(f"voxels differing from the original value at their centre, all volumes: {totalBad:,}")
    ok = shapeOk and affineOk and countOk and totalBad == 0
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
