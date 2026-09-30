"""
Share of each participant's projected signal that falls on mask voxels their scan did not sample.

IN   proj | asso | wb | comm          priors set
     mask                             stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py (03_greymatter_mask)
     n_sample                         optional, default 500
     priors/<set file>.h5             Functionnectome priors, file names in CONFIG
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     stopsignal/functionnectome/input/<sub>_support.nii.gz   made by build_funtome_input.py (02_template_and_route)
     subjects_stopsignal_all.txt   one participant ID per line, in this folder

OUT  logs/missing_signal_share_<set>_<mask folder>_<mask stem>.csv
"""

import csv
import os
import sys
import time

# Must precede the numpy import, as in functionnectome.py
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import h5py
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
PRIORS_DIR = os.path.join(PROJECT_ROOT, "priors")
TEMPLATE = os.path.join(PRIORS_DIR, "priors_template.nii.gz")
SUPPORT_DIR = os.path.join(PROJECT_ROOT, "stopsignal", "functionnectome", "input")
SUPPORT_SUFFIX = "_support.nii.gz"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECT_FILE = os.path.join(SCRIPT_DIR, "subjects_stopsignal_all.txt")

PRIORS_FILE = {
    "proj": "priors_proj_proba_3T.h5",
    "asso": "priors_asso_proba_3T.h5",
    "comm": "priors_comm_proba_3T.h5",
    "wb": "priors_full_proba_3T_comp9_thr0p01.h5",
}
H5_GROUP = "tract_voxel"
KEY = "{}_{}_{}_vox"

N_SAMPLE = 500
SEED = 0
PERCENTILES = [50, 95]
HEAVY = 0.50          # a sampled voxel more than this fabricated for a participant
SHOW = 30
# ====================================================================


def stem(path):
    full = os.path.abspath(path)
    base = os.path.basename(full)
    for suffix in (".nii.gz", ".nii"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.basename(os.path.dirname(full)) + "_" + base


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("give the priors set and the mask")
    setLabel, maskPath = sys.argv[1:3]
    if setLabel not in PRIORS_FILE:
        sys.exit(f"unknown set {setLabel!r}; give one of " + ", ".join(sorted(PRIORS_FILE)))
    nSample = int(sys.argv[3]) if len(sys.argv) > 3 else N_SAMPLE

    h5path = os.path.join(PRIORS_DIR, PRIORS_FILE[setLabel])
    tpl = nib.load(TEMPLATE)
    tplData = np.asanyarray(tpl.dataobj)
    maskImg = nib.load(maskPath)
    mask = np.asanyarray(maskImg.dataobj) > 0
    if mask.shape != tplData.shape or not np.allclose(maskImg.affine, tpl.affine):
        sys.exit("ERROR: the mask is not on the priors grid")

    subs = [l.strip() for l in open(SUBJECT_FILE) if l.strip()]
    print(f"set:      {setLabel}")
    print(f"priors:   {h5path}")
    print(f"mask:     {maskPath}   {int(mask.sum()):,} voxels")
    print(f"subjects: {len(subs)}")

    t0 = time.time()
    notCovered = np.zeros((len(subs), int(mask.sum())), dtype=np.float32)
    used = []
    for n, sub in enumerate(subs):
        f = os.path.join(SUPPORT_DIR, sub + SUPPORT_SUFFIX)
        if not os.path.isfile(f):
            print(f"  no support file for {sub}, left out")
            continue
        img = nib.load(f)
        if img.shape[:3] != mask.shape or not np.allclose(img.affine, tpl.affine):
            sys.exit(f"ERROR: {f} is not on the priors grid")
        support = np.asanyarray(img.dataobj) > 0
        notCovered[len(used)] = (~support)[mask].astype(np.float32)
        used.append(sub)
    notCovered = notCovered[:len(used)]
    print(f"support loaded for {len(used)} participants in {time.time() - t0:.0f} s\n")

    idx = np.argwhere(tplData > 0)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(idx), size=min(nSample, len(idx)), replace=False)
    sample = idx[pick]
    print(f"sampling: {len(sample):,} template voxels, seed {SEED}\n")

    shares = []
    totals = []
    t0 = time.time()
    with h5py.File(h5path, "r") as h5:
        grp = h5[H5_GROUP]
        for n, (i, j, k) in enumerate(sample, 1):
            key = KEY.format(i, j, k)
            if key not in grp:
                sys.exit(f"no map stored for template voxel {i},{j},{k}")
            w = np.asarray(grp[key][:], dtype=np.float32)[mask]
            total = float(w.sum())
            if total <= 0:
                continue
            shares.append((notCovered @ w) / total)
            totals.append(total)
            if n % 100 == 0:
                print(f"  {n}/{len(sample)} sampled, {time.time() - t0:.0f} s")

    if not shares:
        sys.exit("no sampled voxel draws any weight from inside the mask")
    S = np.vstack(shares).T                      # participants x voxels
    W = np.asarray(totals, dtype=np.float64)     # weight of each sampled voxel
    print(f"\n{S.shape[1]:,} of {len(sample):,} sampled voxels draw weight from the mask\n")

    overall = (S * W).sum(axis=1) / W.sum()
    med, p95 = np.percentile(S, PERCENTILES, axis=1)
    worst = S.max(axis=1)
    heavy = (S > HEAVY).sum(axis=1)

    order = np.argsort(-overall)
    print(f"{'subject':<12}{'overall':>10}{'median':>10}{'p95':>10}{'worst voxel':>14}"
          f"{'voxels >' + f'{HEAVY:.0%}':>14}")
    for r in order[:SHOW]:
        if overall[r] <= 0:
            break
        print(f"{used[r]:<12}{100 * overall[r]:>9.3f}%{100 * med[r]:>9.3f}%"
              f"{100 * p95[r]:>9.3f}%{100 * worst[r]:>13.1f}%{heavy[r]:>14,}")

    nAny = int((overall > 0).sum())
    print(f"\nparticipants with any fabricated signal: {nAny} of {len(used)}")
    print(f"largest overall share: {100 * overall.max():.3f}%")

    os.makedirs(LOG_DIR, exist_ok=True)
    outCsv = os.path.join(LOG_DIR,
                          f"missing_signal_share_{setLabel}_{stem(maskPath)}.csv")
    with open(outCsv, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["subject", "overall_share", "median_share", "p95_share",
                      "worst_voxel_share", f"voxels_over_{HEAVY}"])
        for r in range(len(used)):
            wtr.writerow([used[r], round(float(overall[r]), 8), round(float(med[r]), 8),
                          round(float(p95[r]), 8), round(float(worst[r]), 6),
                          int(heavy[r])])
    print(f"wrote {outCsv}")


if __name__ == "__main__":
    main()
