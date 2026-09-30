"""
Which XTRACT atlas volume each tract protocol falls in, against the label the atlas XML gives it.

IN   atlas, xml                       data/atlases/XTRACT/xtract-tract-atlases-prob-1mm.nii.gz, data/atlases/XTRACT.xml
                                      from FSL's fsl-data_atlases_xtract 1.2.0 or 2.0.0
     protocol folder                  HUMAN/ of the XTRACT team's xtract_data repository, commit b7723aa
     out csv                          labels_<release>.csv

OUT  out csv, one row per protocol found in the atlas labels
"""

import csv
import os
import re
import sys

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
ABBREVIATION_TO_NAME = {
    "ac": "Anterior Commissure",
    "af": "Arcuate Fasciculus",
    "amf": "Amygdalofugal Tract",
    "ar": "Acoustic Radiation",
    "atr": "Anterior Thalamic Radiation",
    "cbd": "Cingulum subsection: Dorsal",
    "cbp": "Cingulum subsection: Peri-genual",
    "cbt": "Cingulum subsection: Temporal",
    "cst": "Corticospinal Tract",
    "emcf": "Extreme Capsule: Frontal",
    "emcp": "Extreme Capsule: Parietal",
    "emct": "Extreme Capsule: Temporal",
    "fa": "Frontal Aslant Tract",
    "fma": "Forceps Major",
    "fmi": "Forceps Minor",
    "fx": "Fornix",
    "ifo": "Inferior Fronto-Occipital Fasciculus",
    "ilf": "Inferior Longitudinal Fasciculus",
    "mb": "Muratoff Bundle",
    "mcp": "Middle Cerebellar Peduncle",
    "mdlf": "Middle Longitudinal Fasciculus",
    "or": "Optic Radiation",
    "slf1": "Superior Longitudinal Fasciculus 1",
    "slf2": "Superior Longitudinal Fasciculus 2",
    "slf3": "Superior Longitudinal Fasciculus 3",
    "str": "Superior Thalamic Radiation",
    "stbf": "Striatal Bundle/External Capsule: Frontal",
    "stbm": "Striatal Bundle/External Capsule: Motor",
    "stbp": "Striatal Bundle/External Capsule: Parietal",
    "stbt": "Striatal Bundle/External Capsule: Temporal",
    "uf": "Uncinate Fasciculus",
    "vof": "Vertical Occipital Fasciculus",
}
SIDE_SUFFIX = {"_l": " L", "_r": " R"}
REGION_FILES = ("seed.nii.gz", "target*.nii.gz")
# ====================================================================


def read_labels(xmlPath):
    text = open(xmlPath, encoding="utf-8", errors="replace").read()
    pairs = re.findall(r'<label[^>]*index="(\d+)"[^>]*>([^<]+)</label>', text)
    idx = [int(i) for i, _ in pairs]
    if idx != list(range(len(idx))):
        sys.exit(f"label indices in {xmlPath} are not 0..n-1 in order")
    return [name.strip() for _, name in pairs]


def protocol_name(folder):
    for suffix, side in SIDE_SUFFIX.items():
        if folder.endswith(suffix):
            base = ABBREVIATION_TO_NAME.get(folder[: -len(suffix)])
            return base + side if base else None
    return ABBREVIATION_TO_NAME.get(folder)


def region_indices(protoDir, shape, affine):
    import glob
    files = []
    for pattern in REGION_FILES:
        files += sorted(glob.glob(os.path.join(protoDir, pattern)))
    regions = []
    for f in files:
        img = nib.load(f)
        vox = np.argwhere(np.asanyarray(img.dataobj) > 0)
        if not np.allclose(img.affine, affine):
            toAtlas = np.linalg.inv(affine) @ img.affine
            shift = toAtlas[:3, 3]
            if not np.allclose(toAtlas[:3, :3], np.eye(3)) or not np.allclose(shift, np.round(shift)):
                sys.exit(f"ERROR: {f} is not on the atlas grid and not a whole-voxel shift of it")
            print(f"  note: {f} header differs from the atlas by a whole-voxel shift {np.round(shift).astype(int)}; "
                  f"placed by its own header")
            vox = vox + np.round(shift).astype(int)
            vox = vox[np.all((vox >= 0) & (vox < np.array(shape)), axis=1)]
        idx = np.ravel_multi_index(vox.T, shape) if len(vox) else np.array([], dtype=int)
        if idx.size == 0:
            return None, []
        regions.append(idx)
    if not regions:
        return None, []
    return regions, [os.path.basename(f) for f in files]


def main():
    if len(sys.argv) != 5:
        sys.exit("give the atlas image, its XML, the protocol folder and the output csv")
    atlasPath, xmlPath, protoRoot, outCsv = sys.argv[1:5]
    for p in (atlasPath, xmlPath, protoRoot):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    labels = read_labels(xmlPath)
    atlas = nib.load(atlasPath)
    shape = atlas.shape[:3]
    nVol = atlas.shape[3]
    print(f"atlas:     {atlasPath}  {nVol} volumes")
    print(f"labels:    {xmlPath}  {len(labels)} labels")
    print(f"protocols: {protoRoot}")
    if nVol != len(labels):
        sys.exit("ERROR: volume count and label count differ")

    protocols = []
    skipped = []
    for folder in sorted(os.listdir(protoRoot)):
        full = os.path.join(protoRoot, folder)
        if not os.path.isdir(full):
            continue
        name = protocol_name(folder)
        if name is None:
            skipped.append(f"{folder} (abbreviation not in CONFIG)")
            continue
        if name not in labels:
            skipped.append(f"{folder} ({name} not in the atlas labels)")
            continue
        idx, files = region_indices(full, shape, atlas.affine)
        if idx is None:
            skipped.append(f"{folder} (no seed or target, or one of them empty)")
            continue
        protocols.append((folder, name, idx, files))
    print(f"protocols matched to a label: {len(protocols)}; skipped: {len(skipped)}")
    for s in skipped:
        print(f"  skipped {s}")

    means = np.zeros((len(protocols), nVol))
    for v in range(nVol):
        vol = np.asanyarray(atlas.dataobj[..., v], dtype=np.float64).ravel()
        for p, (_, _, regions, _) in enumerate(protocols):
            means[p, v] = min(vol[r].mean() for r in regions)

    rows = []
    counts = {"ok": 0, "MISMATCH": 0, "UNDETERMINED": 0}
    print(f"\n{'protocol':<10}{'label of its best volume':<46}{'best':>8}{'2nd':>8}{'own':>8}  status")
    for p, (folder, name, idx, files) in enumerate(protocols):
        order = np.argsort(-means[p])
        best, second = int(order[0]), int(order[1])
        own = labels.index(name)
        if means[p, best] <= 0 or means[p, best] == means[p, second]:
            status = "UNDETERMINED"
        elif best == own:
            status = "ok"
        else:
            status = "MISMATCH"
        counts[status] += 1
        rows.append([folder, name, own, best, labels[best], round(means[p, best], 6),
                     labels[second], round(means[p, second], 6), round(means[p, own], 6),
                     status, " ".join(files), sum(r.size for r in idx)])
        note = f", label {name}" if status == "MISMATCH" else ""
        print(f"{folder:<10}{labels[best]:<46}{means[p, best]:>8.4f}{means[p, second]:>8.4f}"
              f"{means[p, own]:>8.4f}  {status}{note}")

    print(f"\n{counts['ok']} ok, {counts['MISMATCH']} mismatches, {counts['UNDETERMINED']} undetermined "
          f"(best score zero or tied), of {len(protocols)} protocols")
    os.makedirs(os.path.dirname(os.path.abspath(outCsv)), exist_ok=True)
    with open(outCsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "label_name", "label_volume", "best_volume", "best_volume_label",
                    "best_score", "second_volume_label", "second_score", "own_volume_score", "status",
                    "region_files", "region_voxels"])
        w.writerows(rows)
    print(f"wrote {outCsv}")


if __name__ == "__main__":
    main()
