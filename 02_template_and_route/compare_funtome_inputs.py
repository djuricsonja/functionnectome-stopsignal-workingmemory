"""
Compare Functionnectome inputs built with different interpolators.

IN   subject ...                      one or more participant IDs
     stopsignal/functionnectome/input_<interpolator>/<sub>.nii.gz, <sub>_support.nii.gz
                                      made by build_funtome_input.py <sub> <interpolator> input_<interpolator>
     stopsignal/masks/group_aseg/group_gm_mask.nii.gz   made by write_group_gm_mask_aseg.py (03_greymatter_mask)

OUT  printed per participant
"""

import os
import sys
import time

import numpy as np
import nibabel as nib

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
FUNTOME = os.path.join(PROJECT_ROOT, "stopsignal", "functionnectome")
GM_MASK = os.path.join(PROJECT_ROOT, "stopsignal", "masks", "group_aseg", "group_gm_mask.nii.gz")

INTERPOLATORS = ("linear", "bSpline", "lanczosWindowedSinc")
# ====================================================================


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: compare_funtome_inputs.py <subject> [<subject> ...]")
    subs = sys.argv[1:]
    if not os.path.isfile(GM_MASK):
        sys.exit(f"ERROR: not found: {GM_MASK}")
    gm = np.asanyarray(nib.load(GM_MASK).dataobj).astype(bool)
    print(f"group grey-matter mask: {int(gm.sum())} voxels", flush=True)

    for sub in subs:
        print(f"\n================ {sub} ================", flush=True)
        sup_f = os.path.join(FUNTOME, f"input_{INTERPOLATORS[0]}", f"{sub}_support.nii.gz")
        if not os.path.isfile(sup_f):
            print(f"  SKIPPED - no support mask at {sup_f}")
            continue
        support = np.asanyarray(nib.load(sup_f).dataobj) > 0.5
        gm_unsupported = np.logical_and(gm, ~support)
        print(f"  sampled extent {int(support.sum())} voxels; grey-matter voxels "
              f"outside it {int(gm_unsupported.sum())} of {int(gm.sum())} "
              f"({100 * gm_unsupported.sum() / gm.sum():.3f}%)", flush=True)

        vols = {}
        paths = {i: os.path.join(FUNTOME, f"input_{i}", f"{sub}.nii.gz")
                 for i in INTERPOLATORS}
        for name, f in paths.items():
            if not os.path.isfile(f):
                print(f"  {name}: missing, skipped", flush=True)
                continue
            print(f"  reading {name} ({os.path.getsize(f) / 1e6:.0f} MB)...",
                  end="", flush=True)
            t0 = time.time()
            vols[name] = np.asanyarray(nib.load(f).dataobj, dtype=np.float32)
            print(f" {time.time() - t0:.0f} s", flush=True)

        print("  -- leakage into the grey-matter mask --", flush=True)
        for name in INTERPOLATORS:
            if name not in vols:
                continue
            leak = np.logical_and(vols[name].any(axis=3), gm_unsupported)
            n = int(leak.sum())
            line = f"    {name:20s} fabricated in {n:6d} mask voxels"
            if n:
                line += f", largest |value| {np.abs(vols[name][leak]).max():.1f}"
            print(line, flush=True)

        print("  -- how large is a kernel change --", flush=True)
        names = list(vols)
        if not names:
            continue
        scale = float(np.abs(vols[names[0]][support]).mean())
        gm_scale = float(np.abs(vols[names[0]][gm]).mean())
        print(f"    mean |signal|: {scale:.0f} in support, "
              f"{gm_scale:.0f} in grey matter", flush=True)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                x, y = vols[names[a]], vols[names[b]]
                if x.shape != y.shape:
                    print(f"    {names[a]} vs {names[b]}: shapes differ", flush=True)
                    continue
                d = np.abs(x - y)
                s, g = d[support].mean(), d[gm].mean()
                print(f"    {names[a]:20s} vs {names[b]:20s}  max {d.max():9.1f}  "
                      f"support {s:7.2f} ({100 * s / scale:5.2f}%)  "
                      f"grey matter {g:7.2f} ({100 * g / gm_scale:5.2f}%)", flush=True)
                d = None
        vols = None


if __name__ == "__main__":
    main()
