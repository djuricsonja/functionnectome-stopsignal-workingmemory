"""
Share of each participant's projected signal that is not from their own data, over the voxels of the tract
regression, per voxel and per tract.

IN   task                             stopsignal
     set                              proj, asso, wb or comm
     contrast                         SuccStop_vs_Go, UnsuccStop_vs_Go or SuccStop_vs_UnsuccStop
     grey-matter mask                 stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py (03_greymatter_mask)
     white-matter mask                stopsignal/masks/group_aseg/group_wm_mask.nii.gz, made by write_group_wm_mask.py
     workers                          parallel processes
     project root                     optional, default $PROJECT_ROOT
     priors/<set file>.h5             Functionnectome priors, file names in measure_missing_signal_share.py (04_coverage)
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     <task>/functionnectome/<set>/group/<contrast>/mask.nii   made by group_level_stop_signal.m <set> (01_greymatter_glm)
     <task>/functionnectome/input/<sub>_support.nii.gz        made by build_funtome_input.py (02_template_and_route)
     subjects_<task>_all.txt          one participant ID per line, in this folder
     XTRACT image, volume table, JHU labels   as in measure_regression_voxels.py

OUT  logs/missing_signal_share_by_tract_<task>_<set>_<contrast>_<wm stem>.csv   one row per participant and tract
     logs/missing_signal_share_by_voxel_<task>_<set>_<contrast>_<wm stem>.npz   shares (voxels x participants), ijk, subjects
"""

import csv
import os
import sys
import time

# Must precede the numpy import, as in functionnectome.py
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import h5py  # noqa: E402
import nibabel as nib  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_missing_signal_share import PRIORS_FILE, H5_GROUP, KEY  # noqa: E402
from measure_regression_voxels import (XTRACT_IMAGE, XTRACT_TABLE, JHU_IMAGE, JHU_XML, GROUP_MASK,  # noqa: E402
                                       load_img, read_table, read_jhu_indices)

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
PRIORS_DIR = "priors"
TEMPLATE = os.path.join(PRIORS_DIR, "priors_template.nii.gz")
SUPPORT_SUFFIX = "_support.nii.gz"
SUBJECT_FILE = "subjects_{}_all.txt"
LOG_DIR = "logs"
OUT_TRACT = "missing_signal_share_by_tract_{}_{}_{}_{}.csv"
OUT_VOXEL = "missing_signal_share_by_voxel_{}_{}_{}_{}.npz"
CHUNK = 200           # voxels per task handed to a worker
SHOW = 15
# ====================================================================

_notCovered = None
_gm = None
_h5path = None


def init_worker(notCovered, gm, h5path):
    global _notCovered, _gm, _h5path
    _notCovered, _gm, _h5path = notCovered, gm, h5path


def shares_for(ijkChunk):
    out = np.zeros((len(ijkChunk), _notCovered.shape[0]), dtype=np.float32)
    empty = np.zeros(len(ijkChunk), dtype=bool)
    with h5py.File(_h5path, "r") as h5:
        grp = h5[H5_GROUP]
        for n, (i, j, k) in enumerate(ijkChunk):
            key = KEY.format(i, j, k)
            if key not in grp:
                raise KeyError(f"no map stored for voxel {i},{j},{k}")
            w = np.asarray(grp[key][:], dtype=np.float32)[_gm]
            total = float(w.sum())
            if total <= 0:
                empty[n] = True
                continue
            out[n] = (_notCovered @ w) / total
    return out, empty


def check_grid(img, ref, path):
    if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine):
        sys.exit(f"{path}: grid {img.shape[:3]} / affine differs from the priors template")


def main():
    if len(sys.argv) not in (7, 8):
        sys.exit("usage: measure_missing_signal_share_by_tract.py <task> <set> <contrast> <grey-matter mask> "
                 "<white-matter mask> <workers> [project root]")
    task, setLabel, contrast, gmPath, wmPath = sys.argv[1:6]
    workers = int(sys.argv[6])
    root = sys.argv[7] if len(sys.argv) == 8 else DEFAULT_ROOT
    if setLabel not in PRIORS_FILE:
        sys.exit(f"unknown set {setLabel!r}; give one of " + ", ".join(sorted(PRIORS_FILE)))
    wmStem = os.path.basename(wmPath).split(".")[0]
    outTract = os.path.join(root, LOG_DIR, OUT_TRACT.format(task, setLabel, contrast, wmStem))
    outVoxel = os.path.join(root, LOG_DIR, OUT_VOXEL.format(task, setLabel, contrast, wmStem))
    for p in (outTract, outVoxel):
        if os.path.exists(p):
            sys.exit(f"output exists, not overwriting: {p}")

    tpl = load_img(os.path.join(root, TEMPLATE))
    gmImg = load_img(gmPath)
    check_grid(gmImg, tpl, gmPath)
    gm = np.asanyarray(gmImg.dataobj) > 0
    wmImg = load_img(wmPath)
    check_grid(wmImg, tpl, wmPath)
    groupPath = os.path.join(root, task, "functionnectome", setLabel, "group", contrast, GROUP_MASK)
    grpImg = load_img(groupPath)
    check_grid(grpImg, tpl, groupPath)
    voxels = (np.asanyarray(wmImg.dataobj) > 0) & (np.asanyarray(grpImg.dataobj) > 0)
    ijk = np.argwhere(voxels)
    if len(ijk) == 0:
        sys.exit("no voxel in both the group mask and the white-matter mask")

    subFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), SUBJECT_FILE.format(task))
    if not os.path.isfile(subFile):
        sys.exit(f"not found: {subFile}")
    subs = [line.strip() for line in open(subFile) if line.strip()]
    supportDir = os.path.join(root, task, "functionnectome", "input")
    notCovered = np.zeros((len(subs), int(gm.sum())), dtype=np.float32)
    used = []
    for sub in subs:
        f = os.path.join(supportDir, sub + SUPPORT_SUFFIX)
        if not os.path.isfile(f):
            print(f"  no support file for {sub}, left out")
            continue
        img = nib.load(f)
        check_grid(img, tpl, f)
        notCovered[len(used)] = (~(np.asanyarray(img.dataobj) > 0))[gm]
        used.append(sub)
    notCovered = notCovered[:len(used)]

    h5path = os.path.join(root, PRIORS_DIR, PRIORS_FILE[setLabel])
    if not os.path.isfile(h5path):
        sys.exit(f"not found: {h5path}")
    print(f"task {task}   set {setLabel}   contrast {contrast}")
    print(f"priors            {h5path}")
    print(f"grey-matter mask  {gmPath}   {int(gm.sum()):,} voxels")
    print(f"white-matter mask {wmPath}   {int((np.asanyarray(wmImg.dataobj) > 0).sum()):,} voxels")
    print(f"group mask        {groupPath}   {int((np.asanyarray(grpImg.dataobj) > 0).sum()):,} voxels")
    print(f"voxels measured   {len(ijk):,}")
    print(f"participants      {len(used)} of {len(subs)}")
    print(f"workers           {workers}\n")

    chunks = [ijk[a:a + CHUNK] for a in range(0, len(ijk), CHUNK)]
    t0 = time.time()
    parts = []
    with Pool(workers, initializer=init_worker, initargs=(notCovered, gm, h5path)) as pool:
        for n, part in enumerate(pool.imap(shares_for, chunks), 1):
            parts.append(part)
            if n % max(1, len(chunks) // 20) == 0 or n == len(chunks):
                print(f"  {min(n * CHUNK, len(ijk)):,}/{len(ijk):,} voxels, {time.time() - t0:.0f} s")
    S = np.vstack([p[0] for p in parts])          # voxels x participants
    empty = np.concatenate([p[1] for p in parts])
    print(f"\nvoxels drawing no weight from the grey-matter mask: {int(empty.sum())} (left out)")
    keep = ~empty
    S, ijk = S[keep], ijk[keep]
    sel = tuple(ijk.T)

    xtImg = load_img(os.path.join(root, XTRACT_IMAGE))
    check_grid(xtImg, tpl, XTRACT_IMAGE)
    names = read_table(os.path.join(root, XTRACT_TABLE))
    if sorted(names) != list(range(xtImg.shape[3])):
        sys.exit(f"table volumes do not match the image's {xtImg.shape[3]} volumes")
    xt = np.asanyarray(xtImg.dataobj)
    tracts = [(names[v], "XTRACT", xt[..., v][sel].astype(np.float64)) for v in sorted(names)]
    jhuImg = load_img(JHU_IMAGE)
    check_grid(jhuImg, tpl, JHU_IMAGE)
    jhu = np.asanyarray(jhuImg.dataobj).astype(np.int64)
    for name, val in read_jhu_indices(JHU_XML).items():
        tracts.append((name, "JHU", (jhu[sel] == val).astype(np.float64)))

    rows = []
    T = np.zeros((len(tracts), len(used)))
    for t, (name, src, p) in enumerate(tracts):
        if p.sum() <= 0:
            sys.exit(f"tract {name} has no voxel among those measured")
        T[t] = (p @ S) / p.sum()
        for r, sub in enumerate(used):
            rows.append([sub, name, src, int((p > 0).sum()), round(float(T[t, r]), 8)])

    print(f"\nper tract, worst participant first ({min(SHOW, len(tracts))} of {len(tracts)} tracts)")
    print(f"{'tract':<48}{'voxels':>8}{'worst':>10}  {'participant':<12}{'participants > 0':>18}")
    worst = T.max(axis=1)
    for t in np.argsort(-worst)[:SHOW]:
        name, src, p = tracts[t]
        print(f"{name:<48}{int((p > 0).sum()):>8}{100 * worst[t]:>9.3f}%  {used[int(T[t].argmax())]:<12}"
              f"{int((T[t] > 0).sum()):>18}")

    print(f"\nper participant, largest tract share first ({min(SHOW, len(used))} of {len(used)})")
    print(f"{'subject':<12}{'largest tract share':>20}  {'tract':<48}{'mean over voxels':>18}")
    subWorst = T.max(axis=0)
    voxMean = S.mean(axis=0)
    for r in np.argsort(-subWorst)[:SHOW]:
        print(f"{used[r]:<12}{100 * subWorst[r]:>19.3f}%  {tracts[int(T[:, r].argmax())][0]:<48}"
              f"{100 * voxMean[r]:>17.3f}%")
    print(f"\nparticipants with any tract share above 0: {int((subWorst > 0).sum())} of {len(used)}")

    os.makedirs(os.path.dirname(outTract), exist_ok=True)
    with open(outTract, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "tract", "source", "voxels", "share"])
        w.writerows(rows)
    np.savez_compressed(outVoxel, shares=S, ijk=ijk, subjects=np.array(used))
    print(f"wrote {outTract}")
    print(f"wrote {outVoxel}")


if __name__ == "__main__":
    main()
