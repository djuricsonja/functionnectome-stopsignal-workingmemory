"""
Match every tract map of one XTRACT atlas release against every tract map of another.

IN   old atlas, old xml, old scale    data/atlases/XTRACT/xtract-tract-atlases-prob-1mm.nii.gz, data/atlases/XTRACT.xml, 100
                                      from FSL's fsl-data_atlases_xtract 1.2.0
     new atlas, new xml, new scale    the same two files from fsl-data_atlases_xtract 2.0.0, 1
     out prefix                       compare_1.2.0_2.0.0
     scale                            the value meaning 100% in that file

OUT  <out prefix>_correlation.csv, _dice.csv, _matches.csv, _position.csv
"""

import csv
import os
import re
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
FLOOR = 0.05
# ====================================================================


def read_labels(xmlPath):
    text = open(xmlPath, encoding="utf-8", errors="replace").read()
    pairs = re.findall(r'<label[^>]*index="(\d+)"[^>]*>([^<]+)</label>', text)
    if [int(i) for i, _ in pairs] != list(range(len(pairs))):
        sys.exit(f"label indices in {xmlPath} are not 0..n-1 in order")
    return [name.strip() for _, name in pairs]


def load(atlasPath, xmlPath, scale):
    img = nib.load(atlasPath)
    labels = read_labels(xmlPath)
    if img.shape[3] != len(labels):
        sys.exit(f"{atlasPath}: {img.shape[3]} volumes but {len(labels)} labels")
    return img, labels, float(scale)


def volume(img, v, scale):
    return np.asanyarray(img.dataobj[..., v], dtype=np.float32) / scale


def main():
    if len(sys.argv) != 8:
        sys.exit("give old atlas, old xml, old scale, new atlas, new xml, new scale, output prefix")
    oldImg, oldLab, oldScale = load(*sys.argv[1:4])
    newImg, newLab, newScale = load(*sys.argv[4:7])
    prefix = sys.argv[7]
    if oldImg.shape[:3] != newImg.shape[:3] or not np.allclose(oldImg.affine, newImg.affine):
        sys.exit("ERROR: the two atlases are not on the same grid")
    shape = oldImg.shape[:3]
    print(f"old: {sys.argv[1]}  {len(oldLab)} maps, scale {oldScale:g}")
    print(f"new: {sys.argv[4]}  {len(newLab)} maps, scale {newScale:g}")
    print(f"grid {shape}, floor for Dice and extent {FLOOR:g}")

    domain = np.zeros(shape, dtype=bool)
    for img, labs, sc in ((oldImg, oldLab, oldScale), (newImg, newLab, newScale)):
        for v in range(len(labs)):
            domain |= volume(img, v, sc) > 0
    dIdx = np.flatnonzero(domain)
    mm = nib.affines.apply_affine(oldImg.affine, np.argwhere(domain))
    print(f"domain: {dIdx.size:,} voxels non-zero in at least one map")

    positions = []

    def extract(img, labs, sc, tag):
        X = np.zeros((dIdx.size, len(labs)), dtype=np.float32)
        for v in range(len(labs)):
            vol = volume(img, v, sc).ravel()[dIdx]
            X[:, v] = vol
            w = vol.astype(np.float64)
            centre = (mm * w[:, None]).sum(0) / w.sum()
            body = mm[vol >= FLOOR]
            lo = body.min(0) if len(body) else [np.nan] * 3
            hi = body.max(0) if len(body) else [np.nan] * 3
            positions.append([tag, v, labs[v]] + [round(c, 1) for c in centre] +
                             [f"{lo[a]:.0f}..{hi[a]:.0f}" for a in range(3)] +
                             [int((vol > 0).sum()), int((vol >= FLOOR).sum()), round(float(vol.max()), 4)])
        return X

    A = extract(oldImg, oldLab, oldScale, "old")
    B = extract(newImg, newLab, newScale, "new")

    Ab = (A >= FLOOR).astype(np.float32)
    Bb = (B >= FLOOR).astype(np.float32)
    D = 2 * (Ab.T @ Bb).astype(np.float64) / (Ab.sum(0, dtype=np.float64)[:, None] +
                                              Bb.sum(0, dtype=np.float64)[None, :])
    del Ab, Bb
    for X in (A, B):
        X -= X.mean(0, dtype=np.float64).astype(np.float32)
        X /= X.std(0, dtype=np.float64).astype(np.float32)
    R = (A.T @ B).astype(np.float64) / dIdx.size

    for name, M in (("correlation", R), ("dice", D)):
        with open(f"{prefix}_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["old \\ new"] + newLab)
            for i, lab in enumerate(oldLab):
                w.writerow([lab] + [round(float(x), 4) for x in M[i]])

    rows = []
    nMis = 0
    print(f"\n{'old map':<42}{'r same name':>12}  {'best new map':<46}{'r':>7}{'Dice':>7}  {'2nd, r':<30}")
    for i, lab in enumerate(oldLab):
        order = np.argsort(-R[i])
        b, s = int(order[0]), int(order[1])
        same = newLab.index(lab) if lab in newLab else None
        rSame = R[i, same] if same is not None else np.nan
        status = "same" if same == b else "MISMATCH"
        nMis += status == "MISMATCH"
        rows.append([lab, same, round(float(rSame), 4),
                     round(float(D[i, same]), 4) if same is not None else "",
                     b, newLab[b], round(float(R[i, b]), 4), round(float(D[i, b]), 4),
                     newLab[s], round(float(R[i, s]), 4), status])
        print(f"{lab:<42}{rSame:>12.3f}  {newLab[b]:<46}{R[i, b]:>7.3f}{D[i, b]:>7.3f}  "
              f"{newLab[s][:22]:<23}{R[i, s]:.3f}  {status}")
    print(f"\n{len(oldLab) - nMis} of {len(oldLab)} old maps match best the new map of the same name; "
          f"{nMis} do not")

    with open(f"{prefix}_matches.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["old_map", "new_volume_same_name", "r_same_name", "dice_same_name",
                    "best_new_volume", "best_new_label", "r_best", "dice_best",
                    "second_new_label", "r_second", "status"])
        w.writerows(rows)
    with open(f"{prefix}_position.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "volume", "label", "centre_x_mm", "centre_y_mm", "centre_z_mm",
                    "extent_x_mm", "extent_y_mm", "extent_z_mm",
                    "voxels_nonzero", "voxels_at_floor", "max"])
        w.writerows(positions)
    print(f"wrote {prefix}_correlation.csv, _dice.csv, _matches.csv, _position.csv")


if __name__ == "__main__":
    main()
