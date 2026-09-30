"""
One-sample group t-map computed directly from first-level contrast images, and its check
against SPM's group map for the same set and contrast.

IN   set                              proj, asso, wb or comm
     contrast                         SuccStop_vs_Go, UnsuccStop_vs_Go or SuccStop_vs_UnsuccStop
     project root                     optional, default $PROJECT_ROOT
     stopsignal/functionnectome/<set>/group/<contrast>/{subjects_in_analysis.txt, mask.nii, spmT_0001.nii}
                                      made by group_level_stop_signal.m <set> (01_greymatter_glm)
     stopsignal/functionnectome/<set>/glm/<sub>/con_NNNN.nii, logs/contrast_map_<set>.csv
                                      made by make_contrast_stop_signal.m <set> (01_greymatter_glm)

OUT  logs/group_tmap_check_<set>_<contrast>.csv
"""

import csv
import os
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
FUNTOME_DIR = os.path.join("stopsignal", "functionnectome")
LOG_DIR = "logs"
CONTRAST_MAP = "contrast_map_{}.csv"
SUBJECT_LIST = "subjects_in_analysis.txt"
SPM_T = "spmT_0001.nii"
SPM_MASK = "mask.nii"
OUT_NAME = "group_tmap_check_{}_{}.csv"

SETS = ("proj", "asso", "wb", "comm")
CONTRASTS = ("SuccStop_vs_Go", "UnsuccStop_vs_Go", "SuccStop_vs_UnsuccStop")
# ====================================================================


def load(path):
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    return np.asanyarray(nib.load(path).dataobj, dtype=np.float64)


def read_lines(path):
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def con_paths(root, setLabel, contrast):
    """Subjects and con image paths for one set and contrast, in SPM's subject order."""
    subjects = read_lines(os.path.join(root, FUNTOME_DIR, setLabel, "group", contrast, SUBJECT_LIST))
    mapFile = os.path.join(root, LOG_DIR, CONTRAST_MAP.format(setLabel))
    if not os.path.exists(mapFile):
        sys.exit(f"not found: {mapFile}")
    conNumber = {}
    with open(mapFile, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["contrast"] == contrast:
                if r["subject"] in conNumber:
                    sys.exit(f"{r['subject']} has two rows for {contrast} in {mapFile}")
                conNumber[r["subject"]] = int(r["con_number"])
    missing = [s for s in subjects if s not in conNumber]
    if missing:
        sys.exit(f"in {SUBJECT_LIST} but not in the contrast map: {', '.join(missing)}")
    glm = os.path.join(root, FUNTOME_DIR, setLabel, "glm")
    return subjects, [os.path.join(glm, s, f"con_{conNumber[s]:04d}.nii") for s in subjects]


def load_stack(paths, mask):
    """subjects x voxels array of the con values inside mask (boolean 3D)."""
    stack = np.empty((len(paths), int(mask.sum())), dtype=np.float64)
    for i, p in enumerate(paths):
        img = load(p)
        if img.shape != mask.shape:
            sys.exit(f"{p}: shape {img.shape}, mask {mask.shape}")
        stack[i] = img[mask]
    return stack


def group_t(stack):
    n = stack.shape[0]
    return stack.mean(axis=0) / (stack.std(axis=0, ddof=1) / np.sqrt(n))


def main():
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in SETS or sys.argv[2] not in CONTRASTS:
        sys.exit("usage: group_tmap.py <" + "|".join(SETS) + "> <" + "|".join(CONTRASTS)
                 + "> [project root]")
    setLabel, contrast = sys.argv[1], sys.argv[2]
    root = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_ROOT

    out = os.path.join(root, LOG_DIR, OUT_NAME.format(setLabel, contrast))
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")

    groupDir = os.path.join(root, FUNTOME_DIR, setLabel, "group", contrast)
    spmMask = load(os.path.join(groupDir, SPM_MASK)) > 0
    spmT = load(os.path.join(groupDir, SPM_T))
    subjects, paths = con_paths(root, setLabel, contrast)
    print(f"set {setLabel}   contrast {contrast}   n = {len(subjects)}")
    print(f"group folder {groupDir}")

    allFinite = np.ones(spmMask.shape, dtype=bool)
    allNonzero = np.ones(spmMask.shape, dtype=bool)
    for p in paths:
        img = load(p)
        if img.shape != spmMask.shape:
            sys.exit(f"{p}: shape {img.shape}, SPM mask {spmMask.shape}")
        finite = np.isfinite(img)
        allFinite &= finite
        allNonzero &= finite & (img != 0)

    stack = load_stack(paths, spmMask)
    nonFinite = int((~np.isfinite(stack)).any(axis=0).sum())
    t = group_t(stack)
    ref = spmT[spmMask]
    both = np.isfinite(t) & np.isfinite(ref)
    diff = np.abs(t[both] - ref[both])
    rel = diff / np.maximum(np.abs(ref[both]), np.finfo(np.float32).tiny)
    corr = float(np.corrcoef(t[both], ref[both])[0, 1])

    row = {
        "set": setLabel, "contrast": contrast, "n": len(subjects),
        "spm_mask_voxels": int(spmMask.sum()),
        "all_finite_voxels": int(allFinite.sum()),
        "all_finite_voxels_not_in_spm_mask": int((allFinite & ~spmMask).sum()),
        "spm_mask_voxels_not_all_finite": int((spmMask & ~allFinite).sum()),
        "all_finite_nonzero_voxels": int(allNonzero.sum()),
        "all_finite_nonzero_voxels_not_in_spm_mask": int((allNonzero & ~spmMask).sum()),
        "spm_mask_voxels_not_all_finite_nonzero": int((spmMask & ~allNonzero).sum()),
        "spm_mask_voxels_with_nonfinite_con": nonFinite,
        "voxels_compared": int(both.sum()),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float(rel.max()),
        "mean_abs_diff": float(diff.mean()),
        "correlation": corr,
        "spm_t_min": float(ref[both].min()), "spm_t_max": float(ref[both].max()),
    }
    for k, v in row.items():
        print(f"   {k:<44} {v}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
