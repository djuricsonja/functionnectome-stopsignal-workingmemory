"""
Voxels available to the tract regression: shared by two sets' group masks, with and without the
white-matter mask, and how many of them each tract reaches.

IN   setA, setB                       two of proj, asso, wb, comm
     white-matter mask                stopsignal/masks/group_aseg/group_wm_mask.nii.gz, made by write_group_wm_mask.py
     project root                     optional, default $PROJECT_ROOT
     stopsignal/functionnectome/<set>/group/<contrast>/mask.nii   made by group_level_stop_signal.m <set> (01_greymatter_glm)
     atlases/xtract_2.0.0/xtract-tract-atlases-prob-2mm_flirt_nn.nii.gz   made by resample_atlas_flirt.sh (06_atlas)
     atlases/xtract_2.0.0/xtract_2.0.0_volume_table.csv   made by build_xtract_volume_table.py (06_atlas)
     $FSLDIR/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz, $FSLDIR/data/atlases/JHU-labels.xml

OUT  logs/regression_voxels_<setA>_<setB>_<mask stem>.csv
     logs/regression_tract_voxels_<setA>_<setB>_<mask stem>.csv
"""

import csv
import os
import re
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
FUNTOME_DIR = os.path.join("stopsignal", "functionnectome")
LOG_DIR = "logs"
GROUP_MASK = "mask.nii"
XTRACT_IMAGE = os.path.join("atlases", "xtract_2.0.0", "xtract-tract-atlases-prob-2mm_flirt_nn.nii.gz")
XTRACT_TABLE = os.path.join("atlases", "xtract_2.0.0", "xtract_2.0.0_volume_table.csv")
TABLE_VOLUME_COLUMN = "volume"
TABLE_TRACT_COLUMN = "tract"

FSLDIR = os.environ["FSLDIR"]
JHU_IMAGE = os.path.join(FSLDIR, "data", "atlases", "JHU", "JHU-ICBM-labels-2mm.nii.gz")
JHU_XML = os.path.join(FSLDIR, "data", "atlases", "JHU-labels.xml")
JHU_NAMES = ("Genu of corpus callosum", "Body of corpus callosum", "Splenium of corpus callosum")

SETS = ("proj", "asso", "wb", "comm")
CONTRASTS = ("SuccStop_vs_Go", "UnsuccStop_vs_Go", "SuccStop_vs_UnsuccStop")
OUT_VOXELS = "regression_voxels_{}_{}_{}.csv"
OUT_TRACTS = "regression_tract_voxels_{}_{}_{}.csv"
# ====================================================================


def load_img(path):
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    return nib.load(path)


def same_grid(img, ref, path):
    if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine):
        sys.exit(f"{path}: grid {img.shape[:3]} / affine differs from the white-matter mask")


def read_table(path):
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = {int(r[TABLE_VOLUME_COLUMN]): r[TABLE_TRACT_COLUMN] for r in rows}
    if len(names) != len(rows):
        sys.exit(f"duplicate volume numbers in {path}")
    if len(set(names.values())) != len(names):
        sys.exit(f"duplicate tract names in {path}")
    return names


def read_jhu_indices(path):
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    text = open(path, encoding="utf-8", errors="replace").read()
    labels = [(int(dict(re.findall(r'(\w+)="([^"]*)"', attrs))["index"]), name.strip())
              for attrs, name in re.findall(r"<label([^>]*)>([^<]+)</label>", text)]
    out = {}
    for want in JHU_NAMES:
        hits = [i for i, n in labels if n == want]
        if len(hits) != 1:
            sys.exit(f"'{want}' matches {len(hits)} labels in {path}")
        out[want] = hits[0]
    return out


def main():
    if (len(sys.argv) not in (4, 5) or sys.argv[1] not in SETS or sys.argv[2] not in SETS
            or sys.argv[1] == sys.argv[2]):
        sys.exit("usage: measure_regression_voxels.py <setA> <setB> <white-matter mask> [project root]"
                 "   sets: " + ", ".join(SETS))
    setA, setB, wmPath = sys.argv[1], sys.argv[2], sys.argv[3]
    root = sys.argv[4] if len(sys.argv) == 5 else DEFAULT_ROOT
    stem = os.path.basename(wmPath).split(".")[0]
    outVox = os.path.join(root, LOG_DIR, OUT_VOXELS.format(setA, setB, stem))
    outTr = os.path.join(root, LOG_DIR, OUT_TRACTS.format(setA, setB, stem))
    for p in (outVox, outTr):
        if os.path.exists(p):
            sys.exit(f"output exists, not overwriting: {p}")

    wmImg = load_img(wmPath)
    wm = np.asanyarray(wmImg.dataobj) > 0

    xtImg = load_img(os.path.join(root, XTRACT_IMAGE))
    same_grid(xtImg, wmImg, XTRACT_IMAGE)
    names = read_table(os.path.join(root, XTRACT_TABLE))
    if sorted(names) != list(range(xtImg.shape[3])):
        sys.exit(f"table volumes {min(names)}..{max(names)} ({len(names)}) do not match "
                 f"the image's {xtImg.shape[3]} volumes")
    xt = np.asanyarray(xtImg.dataobj)
    tracts = [(names[v], "XTRACT", v, xt[..., v] > 0) for v in sorted(names)]

    jhuImg = load_img(JHU_IMAGE)
    same_grid(jhuImg, wmImg, JHU_IMAGE)
    jhu = np.asanyarray(jhuImg.dataobj).astype(np.int64)
    for name, idx in read_jhu_indices(JHU_XML).items():
        tracts.append((name, "JHU", idx, jhu == idx))

    print(f"sets {setA} / {setB}   white-matter mask {wmPath} ({int(wm.sum())} voxels)")
    print(f"predictors: {len(tracts)}  ({sum(t[1] == 'XTRACT' for t in tracts)} XTRACT, "
          f"{sum(t[1] == 'JHU' for t in tracts)} JHU; JHU values "
          f"{', '.join(str(t[2]) for t in tracts if t[1] == 'JHU')})\n")

    voxRows, trRows = [], []
    for c in CONTRASTS:
        masks = {}
        for s in (setA, setB):
            p = os.path.join(root, FUNTOME_DIR, s, "group", c, GROUP_MASK)
            img = load_img(p)
            same_grid(img, wmImg, p)
            masks[s] = np.asanyarray(img.dataobj) > 0
        shared = masks[setA] & masks[setB]
        sharedWm = shared & wm
        vr = {"contrast": c, f"{setA}_mask": int(masks[setA].sum()),
              f"{setB}_mask": int(masks[setB].sum()), "shared": int(shared.sum()),
              "wm_mask": int(wm.sum()), "shared_in_wm": int(sharedWm.sum()),
              f"wm_not_in_{setA}": int((wm & ~masks[setA]).sum()),
              f"wm_not_in_{setB}": int((wm & ~masks[setB]).sum())}
        voxRows.append(vr)
        print(f"== {c}")
        for k, v in list(vr.items())[1:]:
            print(f"   {k:<22} {v}")
        print(f"   {'tract':<40} {'source':<7} {'value':>5} {'on grid':>8} {'shared':>8} {'shared in wm':>13}")
        for name, src, idx, m in tracts:
            r = {"contrast": c, "tract": name, "source": src, "volume_or_value": idx,
                 "voxels_on_grid": int(m.sum()), "voxels_shared": int((m & shared).sum()),
                 "voxels_shared_in_wm": int((m & sharedWm).sum())}
            trRows.append(r)
            print(f"   {name:<40} {src:<7} {idx:>5} {r['voxels_on_grid']:>8} {r['voxels_shared']:>8}"
                  f" {r['voxels_shared_in_wm']:>13}")
        zero = [r["tract"] for r in trRows if r["contrast"] == c and r["voxels_shared_in_wm"] == 0]
        print(f"   tracts with no shared white-matter voxel: {len(zero)}"
              + (f"  ({', '.join(zero)})" if zero else "") + "\n")

    for path, rows in ((outVox, voxRows), (outTr, trRows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
