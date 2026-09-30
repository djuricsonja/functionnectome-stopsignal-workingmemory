"""
Whether each coordinate Rae et al. (2015) report is recovered in our group maps.

IN   rae_2015_table1.csv   in this folder
     stopsignal/group_all/<contrast>/{spmT_0001.nii, spmT_0002.nii, mask.nii}
                           made by group_level_stop_signal.m arm1
     stopsignal/group_all/<contrast>/tables/<contrast>_<pos|neg>_p001k0.csv
                           made by export_group_tables.m arm1

OUT  logs/rae_replication.csv        one row per contrast, direction and coordinate
     printed per-region summary
"""

import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
GROUP_ROOT = os.path.join(PROJECT_ROOT, "stopsignal", "group_all")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAE_CSV = os.path.join(SCRIPT_DIR, "rae_2015_table1.csv")
OUT_CSV = os.path.join(LOG_DIR, "rae_replication.csv")

CONTRASTS = ["SuccStop_vs_Go", "UnsuccStop_vs_Go", "SuccStop_vs_UnsuccStop"]
DIRECTIONS = {"pos": "spmT_0001.nii", "neg": "spmT_0002.nii"}

RADIUS = 8.0          # mm
REFERENCE = "SuccStop_vs_Go"
# ====================================================================


def fwe_threshold(table_path):
    with open(table_path) as fh:
        rows = list(csv.reader(fh))
    surviving = []
    for r in rows[2:]:
        if len(r) < 14:
            continue
        try:
            p, t = float(r[6]), float(r[8])
        except ValueError:
            continue
        if p < 0.05:
            surviving.append(t)
    return min(surviving) if surviving else None


def world_grid(affine, shape):
    idx = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    return np.einsum("ij,jklm->iklm", affine[:3, :3], idx) + affine[:3, 3].reshape(3, 1, 1, 1)


def main():
    if not os.path.isfile(RAE_CSV):
        sys.exit(f"ERROR: not found: {RAE_CSV}")
    rae = list(csv.DictReader(open(RAE_CSV)))
    print(f"Rae coordinates: {len(rae)} (from {os.path.basename(RAE_CSV)})")
    print(f"sphere radius: {RADIUS:.0f} mm\n", flush=True)

    rows = []
    for contrast in CONTRASTS:
        d = os.path.join(GROUP_ROOT, contrast)
        mask_f = os.path.join(d, "mask.nii")
        if not os.path.isfile(mask_f):
            print(f"{contrast}: no mask.nii, skipping", flush=True)
            continue
        mask_img = nib.load(mask_f)
        M = np.asanyarray(mask_img.dataobj) > 0
        W = world_grid(mask_img.affine, M.shape)

        for direction, fname in DIRECTIONS.items():
            spmt = os.path.join(d, fname)
            table = os.path.join(d, "tables", f"{contrast}_{direction}_p001k0.csv")
            if not (os.path.isfile(spmt) and os.path.isfile(table)):
                print(f"{contrast} {direction}: missing spmT or table, skipping", flush=True)
                continue
            thr = fwe_threshold(table)
            if thr is None:
                print(f"{contrast} {direction}: nothing survives voxel-FWE, skipping", flush=True)
                continue
            T = np.asanyarray(nib.load(spmt).dataobj)

            n_rec = 0
            for r in rae:
                c = (float(r["x"]), float(r["y"]), float(r["z"]))
                inside = (((W[0] - c[0]) ** 2 + (W[1] - c[1]) ** 2
                           + (W[2] - c[2]) ** 2) <= RADIUS ** 2) & M
                if inside.any():
                    best = float(T[inside].max())
                    j = np.unravel_index(np.argmax(np.where(inside, T, -np.inf)), T.shape)
                    at = mask_img.affine[:3, :3] @ np.array(j) + mask_img.affine[:3, 3]
                    rec = int(best >= thr)
                else:
                    best, at, rec = float("nan"), (np.nan,) * 3, 0
                n_rec += rec
                rows.append(dict(contrast=contrast, direction=direction,
                                 region_as_printed=r["region_as_printed"],
                                 hemisphere=r["hemisphere"],
                                 x=r["x"], y=r["y"], z=r["z"], rae_t=r["t"],
                                 fwe_threshold=round(thr, 2),
                                 our_max_t=round(best, 2) if best == best else "",
                                 our_x=int(at[0]) if at[0] == at[0] else "",
                                 our_y=int(at[1]) if at[1] == at[1] else "",
                                 our_z=int(at[2]) if at[2] == at[2] else "",
                                 recovered=rec))
            print(f"{contrast:24s} {direction:4s} threshold t>={thr:5.2f}   "
                  f"recovered {n_rec:2d} / {len(rae)}", flush=True)

    if not rows:
        sys.exit("ERROR: nothing tested")

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    print(f"per-region recovery for {REFERENCE}, positive direction "
          f"(the contrast closest to Rae's)")
    agg = OrderedDict()
    for r in rows:
        if r["contrast"] == REFERENCE and r["direction"] == "pos":
            a = agg.setdefault(r["region_as_printed"], [0, 0])
            a[0] += r["recovered"]
            a[1] += 1
    for region, (n, total) in agg.items():
        print(f"  {region:44s} {n:2d} / {total}")
    print(f"\nwrote {OUT_CSV}")
    print("=" * 78)


if __name__ == "__main__":
    main()
