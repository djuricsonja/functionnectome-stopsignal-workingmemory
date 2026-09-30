"""
Design for the tract regression: the 63 tract predictors and the group-mean contrast values of two sets, over the
voxels both sets share, optionally restricted to a white-matter mask. One set given twice gives a design for that
set alone.

IN   setA, setB                       proj, asso, wb or comm
     white-matter mask                stopsignal/masks/group_aseg/group_wm_mask.nii.gz, made by write_group_wm_mask.py, or none
     project root                     optional, default $PROJECT_ROOT
     stopsignal/functionnectome/<set>/group/<contrast>/{mask.nii, beta_0001.nii}   made by group_level_stop_signal.m <set> (01_greymatter_glm)
     XTRACT image, volume table, JHU labels   as in measure_regression_voxels.py

OUT  stopsignal/functionnectome/tract_regression/design_<setA>_<setB>_<mask stem | nowm>_mean.mat
       X        voxels x predictors, tract probabilities (JHU: 1 inside the label, 0 outside)
       Y        voxels x sets x contrasts, beta_0001 values
       tract, source, volume_or_value   one entry per predictor
       sets, contrasts, ijk (0-based voxel indices), affine
"""

import os
import sys

import numpy as np
import nibabel as nib
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_regression_voxels import (FUNTOME_DIR, GROUP_MASK, XTRACT_IMAGE, XTRACT_TABLE, JHU_IMAGE,  # noqa: E402
                                       JHU_XML, SETS, CONTRASTS, load_img, read_table, read_jhu_indices)

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
OUT_DIR = os.path.join("stopsignal", "functionnectome", "tract_regression")
OUT_NAME = "design_{}_{}_{}_mean.mat"
GROUP_MAP = "beta_0001.nii"
NO_MASK = "none"
NO_MASK_STEM = "nowm"
# ====================================================================


def check_grid(img, ref, path):
    if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine):
        sys.exit(f"{path}: grid {img.shape[:3]} / affine differs from {ref.get_filename()}")


def main():
    if len(sys.argv) not in (4, 5) or sys.argv[1] not in SETS or sys.argv[2] not in SETS:
        sys.exit("usage: build_regression_design.py <setA> <setB> <white-matter mask | none> [project root]"
                 "   sets: " + ", ".join(SETS))
    setA, setB, wmPath = sys.argv[1], sys.argv[2], sys.argv[3]
    root = sys.argv[4] if len(sys.argv) == 5 else DEFAULT_ROOT
    stem = NO_MASK_STEM if wmPath == NO_MASK else os.path.basename(wmPath).split(".")[0]
    out = os.path.join(root, OUT_DIR, OUT_NAME.format(setA, setB, stem))
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")

    xtImg = load_img(os.path.join(root, XTRACT_IMAGE))
    ref = xtImg
    wm = None
    if wmPath != NO_MASK:
        wmImg = load_img(wmPath)
        check_grid(wmImg, ref, wmPath)
        wm = np.asanyarray(wmImg.dataobj) > 0

    voxels = None
    tvals = {}
    for c in CONTRASTS:
        shared = np.ones(ref.shape[:3], dtype=bool)
        for s in (setA, setB):
            p = os.path.join(root, FUNTOME_DIR, s, "group", c, GROUP_MASK)
            img = load_img(p)
            check_grid(img, ref, p)
            shared &= np.asanyarray(img.dataobj) > 0
        if wm is not None:
            shared &= wm
        if voxels is None:
            voxels = shared
        elif not np.array_equal(voxels, shared):
            sys.exit(f"voxel set differs in {c} from {CONTRASTS[0]}: "
                     f"{int((voxels ^ shared).sum())} voxels")
        for s in (setA, setB):
            p = os.path.join(root, FUNTOME_DIR, s, "group", c, GROUP_MAP)
            img = load_img(p)
            check_grid(img, ref, p)
            tvals[(s, c)] = np.asanyarray(img.dataobj, dtype=np.float64)

    ijk = np.argwhere(voxels)
    sel = tuple(ijk.T)
    Y = np.empty((len(ijk), 2, len(CONTRASTS)))
    for a, s in enumerate((setA, setB)):
        for b, c in enumerate(CONTRASTS):
            Y[:, a, b] = tvals[(s, c)][sel]
    if not np.isfinite(Y).all():
        sys.exit(f"{int((~np.isfinite(Y)).sum())} group values are not finite")

    names = read_table(os.path.join(root, XTRACT_TABLE))
    if sorted(names) != list(range(xtImg.shape[3])):
        sys.exit(f"table volumes do not match the image's {xtImg.shape[3]} volumes")
    xt = np.asanyarray(xtImg.dataobj)
    cols, tract, source, idx = [], [], [], []
    for v in sorted(names):
        cols.append(xt[..., v][sel].astype(np.float64))
        tract.append(names[v]); source.append("XTRACT"); idx.append(v)
    jhuImg = load_img(JHU_IMAGE)
    check_grid(jhuImg, ref, JHU_IMAGE)
    jhu = np.asanyarray(jhuImg.dataobj).astype(np.int64)
    for name, val in read_jhu_indices(JHU_XML).items():
        cols.append((jhu[sel] == val).astype(np.float64))
        tract.append(name); source.append("JHU"); idx.append(val)
    X = np.column_stack(cols)

    empty = [t for t, n in zip(tract, (X > 0).sum(axis=0)) if n == 0]
    if empty:
        sys.exit(f"predictors with no voxel: {', '.join(empty)}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    savemat(out, {"X": X, "Y": Y, "tract": np.array(tract, dtype=object),
                  "source": np.array(source, dtype=object), "volume_or_value": np.array(idx, dtype=float),
                  "sets": np.array([setA, setB], dtype=object),
                  "contrasts": np.array(CONTRASTS, dtype=object),
                  "ijk": ijk.astype(float), "affine": ref.affine}, do_compression=True)
    print(f"sets {setA} / {setB}   mask {wmPath}")
    print(f"voxels {X.shape[0]}   predictors {X.shape[1]}  ({source.count('XTRACT')} XTRACT, {source.count('JHU')} JHU)")
    print(f"voxels per predictor: min {int((X > 0).sum(axis=0).min())}  max {int((X > 0).sum(axis=0).max())}")
    for a, s in enumerate((setA, setB)):
        for b, c in enumerate(CONTRASTS):
            print(f"   {s:<5} {c:<24} group mean over voxels {Y[:, a, b].mean():+.4f}  sd {Y[:, a, b].std():.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
