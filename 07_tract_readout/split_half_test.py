"""
Split-half test of whether two sets' tract patterns differ, for one contrast.

IN   design                           stopsignal/functionnectome/tract_regression/design_<setA>_<setB>_<mask>_mean.mat
                                      made by build_regression_design.py
     contrast                         SuccStop_vs_Go, UnsuccStop_vs_Go or SuccStop_vs_UnsuccStop
     project root                     optional, default $PROJECT_ROOT
     stopsignal/functionnectome/<set>/group/<contrast>/subjects_in_analysis.txt   made by group_level_stop_signal.m <set> (01_greymatter_glm)
     stopsignal/functionnectome/<set>/glm/<sub>/con_NNNN.nii, logs/contrast_map_<set>.csv
                                      made by make_contrast_stop_signal.m <set> (01_greymatter_glm)

OUT  <design folder>/splithalf_<setA>_<setB>_<contrast>_<mask>_mean.csv   one row per split
"""

import csv
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_tmap import FUNTOME_DIR, SUBJECT_LIST, con_paths, load, read_lines  # noqa: E402

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
N_SPLITS = 1000
SEED = 20260927
ALPHA = 0.05
OUT_NAME = "splithalf_{}_{}_{}_{}.csv"
# ====================================================================


def beta_operator(X):
    """Matrix mapping a voxel vector to the tract betas of an OLS fit with intercept."""
    Z = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.pinv(Z)[1:]


def group_mean(stack):
    return stack.mean(axis=0)


def split_half(stackA, stackB, B, n_splits, seed, stat=group_mean):
    """Per-split correlations for two subjects x voxels stacks with rows in the same participant order."""
    n = stackA.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((n_splits, 6))
    for s in range(n_splits):
        perm = rng.permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2:]
        bA1, bA2 = B @ stat(stackA[h1]), B @ stat(stackA[h2])
        bB1, bB2 = B @ stat(stackB[h1]), B @ stat(stackB[h2])
        rAA = np.corrcoef(bA1, bA2)[0, 1]
        rBB = np.corrcoef(bB1, bB2)[0, 1]
        rA1B2 = np.corrcoef(bA1, bB2)[0, 1]
        rA2B1 = np.corrcoef(bA2, bB1)[0, 1]
        prod = rAA * rBB
        out[s] = (rAA, rBB, rA1B2, rA2B1, (rA1B2 + rA2B1) / 2, np.sqrt(prod) if prod >= 0 else np.nan)
    return out


def p_value(res):
    """cross >= sqrt(rAA x rBB), tested as cross >= 0 and cross^2 >= rAA x rBB, so no square root rounds."""
    prod, cross = res[:, 0] * res[:, 1], res[:, 4]
    ok = prod >= 0
    hit = (cross >= 0) & (cross * cross >= prod)
    return float(hit[ok].mean()) if ok.any() else float("nan"), int((~ok).sum())


def read_design(design, contrast):
    """Design contents, set names, contrast names and the design's mask stem."""
    if not os.path.exists(design):
        sys.exit(f"not found: {design}")
    D = loadmat(design)
    sets = [str(np.squeeze(s)).strip() for s in np.ravel(D["sets"])]
    contrasts = [str(np.squeeze(c)).strip() for c in np.ravel(D["contrasts"])]
    if contrast not in contrasts:
        sys.exit(f"{contrast} not in the design's contrasts: {', '.join(contrasts)}")
    stem = os.path.splitext(os.path.basename(design))[0]
    return D, sets, contrasts, stem.split(f"design_{sets[0]}_{sets[1]}_", 1)[1]


def load_stacks(D, sets, contrast, root):
    """One subjects x voxels array per set, rows in the same (sorted) participant order."""
    lists = [read_lines(os.path.join(root, FUNTOME_DIR, s, "group", contrast, SUBJECT_LIST)) for s in sets]
    if sorted(lists[0]) != sorted(lists[1]):
        onlyA = sorted(set(lists[0]) - set(lists[1]))
        onlyB = sorted(set(lists[1]) - set(lists[0]))
        sys.exit(f"participant lists differ: only {sets[0]} {onlyA}, only {sets[1]} {onlyB}")

    ijk = D["ijk"].astype(int)
    stacks = []
    for s in sets:
        subjects, paths = con_paths(root, s, contrast)
        order = np.argsort(subjects)
        first = load(paths[0])
        mask = np.zeros(first.shape, dtype=bool)
        mask[tuple(ijk.T)] = True
        if not np.array_equal(np.argwhere(mask), ijk):
            sys.exit("design voxel indices are not in row-major order")
        stack = np.empty((len(paths), len(ijk)))
        for i, k in enumerate(order):
            stack[i] = load(paths[k])[mask]
        if not np.isfinite(stack).all():
            sys.exit(f"{s}: {int((~np.isfinite(stack)).sum())} non-finite con values at design voxels")
        stacks.append(stack)
    return stacks


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: split_half_test.py <design .mat> <contrast> [project root]")
    design, contrast = sys.argv[1], sys.argv[2]
    root = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_ROOT
    D, sets, contrasts, maskStem = read_design(design, contrast)
    out = os.path.join(os.path.dirname(design), OUT_NAME.format(sets[0], sets[1], contrast, maskStem))
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")

    X = D["X"]
    stacks = load_stacks(D, sets, contrast, root)
    n = stacks[0].shape[0]
    print(f"design {design}\ncontrast {contrast}   sets {sets[0]} / {sets[1]}   n = {n}   halves {n // 2} / {n - n // 2}")
    print(f"voxels {X.shape[0]}   tracts {X.shape[1]}   splits {N_SPLITS}   seed {SEED}")

    b = contrasts.index(contrast)
    for a, s in enumerate(sets):
        d = np.abs(group_mean(stacks[a]) - D["Y"][:, a, b]).max()
        print(f"full-sample mean vs design ({s}): max abs difference {d:.3e}")

    B = beta_operator(X)
    res = split_half(stacks[0], stacks[1], B, N_SPLITS, SEED)
    p, undefined = p_value(res)
    names = [f"r_{sets[0]}_h1_{sets[0]}_h2", f"r_{sets[1]}_h1_{sets[1]}_h2", f"r_{sets[0]}_h1_{sets[1]}_h2",
             f"r_{sets[0]}_h2_{sets[1]}_h1", "cross", "reference"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split"] + names)
        for i, row in enumerate(res):
            w.writerow([i + 1] + [f"{v:.6f}" for v in row])
    med = np.nanmedian(res, axis=0)
    for k, nm in enumerate(names):
        print(f"   median {nm:<28} {med[k]:+.4f}")
    print(f"splits with undefined reference: {undefined}")
    print(f"p = {p:.4f}   ->   {'DISTINGUISHABLE' if p < ALPHA else 'not distinguishable'} (alpha {ALPHA})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
