"""
Estimate the MNI152NLin2009cAsym-to-MNI152NLin6Asym template correction and score it.

IN   <label>                   argument 1, names the output folder xfm_<label>
     [type_of_transform]       argument 2, optional
     [iterations]              argument 3, optional, comma separated
     priors/priors_template.nii.gz        the 'template' dataset of priors_cereb_deter3T.h5, saved as NIfTI
     TemplateFlow MNI152NLin2009cAsym T1w and brain mask at 2 mm (downloaded on first use)

OUT  priors/xfm_<label>/2009cAsym_to_NLin6Asym_{0GenericAffine.mat, 1Warp.nii.gz, 1InverseWarp.nii.gz}
     priors/xfm_<label>/local_r_<name>.nii.gz
     priors/xfm_<label>/jacobian_<name>.nii.gz
     printed comparison
"""

import os
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage

import ants
import templateflow.api as tf

# ============================== CONFIG ==============================
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
PRIORS_TEMPLATE = os.path.join(PROJECT_ROOT, "priors", "priors_template.nii.gz")
PRIORS_DIR = os.path.join(PROJECT_ROOT, "priors")

PREFIX = "2009cAsym_to_NLin6Asym_"
WINDOW = 5          # voxels
# ====================================================================


def local_r(a, b, mask, w):
    k = np.ones((w, w, w), dtype=np.float32)
    n = ndimage.convolve(mask.astype(np.float32), k, mode="constant")
    sa = ndimage.convolve(a * mask, k, mode="constant")
    sb = ndimage.convolve(b * mask, k, mode="constant")
    saa = ndimage.convolve(a * a * mask, k, mode="constant")
    sbb = ndimage.convolve(b * b * mask, k, mode="constant")
    sab = ndimage.convolve(a * b * mask, k, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sab / n - (sa / n) * (sb / n)
        va = saa / n - (sa / n) ** 2
        vb = sbb / n - (sb / n) ** 2
        r = cov / np.sqrt(va * vb)
    r[~np.isfinite(r)] = 0.0
    return r


def grad_mag(x):
    g = np.gradient(x)
    return np.sqrt(sum(c ** 2 for c in g)).astype(np.float32)


def score(slug, name, warped, ref, brain, out_dir):
    a, b = warped[brain], ref[brain]
    r = float(np.corrcoef(a, b)[0, 1])
    ga, gb = grad_mag(warped), grad_mag(ref)
    rg = float(np.corrcoef(ga[brain], gb[brain])[0, 1])
    lr = local_r(warped, ref, brain, WINDOW)
    v = lr[brain]
    nib.save(nib.Nifti1Image(lr.astype(np.float32), nib.load(PRIORS_TEMPLATE).affine),
             os.path.join(out_dir, f"local_r_{slug}.nii.gz"))
    print(f"  {name}")
    print(f"    intensity correlation over the brain   {r:.5f}")
    print(f"    gradient correlation over the brain    {rg:.5f}")
    print(f"    local correlation: median {np.median(v):.4f}, "
          f"10th pct {np.percentile(v, 10):.4f}, "
          f"below 0.5 in {100 * (v < 0.5).mean():.2f}% of brain", flush=True)
    return r, rg, float(np.median(v))


def main():
    if len(sys.argv) not in (2, 3, 4):
        sys.exit("Usage: register_templates.py <label> [<type_of_transform>] "
                 "[<iterations, comma separated>]")
    label = sys.argv[1]
    ttype = sys.argv[2] if len(sys.argv) >= 3 else "SyNCC"
    iters = (tuple(int(x) for x in sys.argv[3].split(","))
             if len(sys.argv) == 4 else (100, 70, 50, 20))
    out_dir = os.path.join(PRIORS_DIR, f"xfm_{label}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"antspyx {ants.__version__}   type_of_transform {ttype}   label {label}   "
          f"reg_iterations {iters}\n", flush=True)

    print("fetching the 2009cAsym template from TemplateFlow...", flush=True)
    t1 = str(tf.get("MNI152NLin2009cAsym", resolution=2, suffix="T1w",
                    desc=None, extension="nii.gz"))
    mk = str(tf.get("MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="mask",
                    extension="nii.gz"))
    print(f"  T1w  {t1}\n  mask {mk}", flush=True)

    fixed = ants.image_read(PRIORS_TEMPLATE)
    moving = ants.image_read(t1) * ants.image_read(mk)
    print(f"fixed {fixed.shape}  moving {moving.shape}", flush=True)

    new_w = os.path.join(out_dir, PREFIX + "1Warp.nii.gz")
    new_a = os.path.join(out_dir, PREFIX + "0GenericAffine.mat")
    if os.path.isfile(new_w) and os.path.isfile(new_a):
        print(f"transform already present in {out_dir}, reusing it", flush=True)
        fwd = [new_w, new_a]
    else:
        print(f"registering with {ttype}, iterations {iters}...", flush=True)
        t0 = time.time()
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=ttype,
                                reg_iterations=iters,
                                outprefix=os.path.join(out_dir, PREFIX), verbose=True)
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
        fwd = reg["fwdtransforms"]
    print(f"  forward transforms: {[os.path.basename(x) for x in fwd]}", flush=True)

    ref = fixed.numpy().astype(np.float32)
    brain = ref > 0
    print(f"\nscoring inside the priors brain, {int(brain.sum())} voxels, "
          f"window {WINDOW} voxels\n", flush=True)

    runs = [("uncorrected", "uncorrected (resampled only)", []),
            ("new", f"new ({ttype}, {'/'.join(str(i) for i in iters)})", fwd)]

    results = {}
    for slug, name, tlist in runs:
        w = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tlist,
                                  interpolator="linear").numpy().astype(np.float32)
        if w.shape != ref.shape:
            sys.exit(f"ERROR: {name} came back as {w.shape}, expected {ref.shape}")
        results[name] = score(slug, name, w, ref, brain, out_dir)

    print("\nfolding check on the deformation fields (Jacobian determinant):", flush=True)
    for slug, name, tlist in runs:
        warp = next((t for t in tlist if t.endswith("Warp.nii.gz")), None)
        if warp is None:
            print(f"  {name}: no deformation field")
            continue
        jac = ants.create_jacobian_determinant_image(fixed, warp, do_log=False).numpy()
        v = jac[brain]
        neg = int((v <= 0).sum())
        print(f"  {name}")
        print(f"    min {v.min():.4f}, 1st pct {np.percentile(v, 1):.4f}, "
              f"median {np.median(v):.4f}, 99th pct {np.percentile(v, 99):.4f}, "
              f"max {v.max():.4f}")
        print(f"    folded voxels (determinant <= 0): {neg} of {int(brain.sum())} "
              f"({100 * neg / brain.sum():.4f}%)"
              f"{'   <-- FOLDING' if neg else ''}", flush=True)
        nib.save(nib.Nifti1Image(jac.astype(np.float32),
                                 nib.load(PRIORS_TEMPLATE).affine),
                 os.path.join(out_dir, f"jacobian_{slug}.nii.gz"))

    print("\nsummary (higher is better):")
    print(f"  {'transform':32s} {'intensity r':>12s} {'gradient r':>12s} {'median local r':>15s}")
    for name, (r, rg, m) in results.items():
        print(f"  {name:32s} {r:12.5f} {rg:12.5f} {m:15.4f}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
