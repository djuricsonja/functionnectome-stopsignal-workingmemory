"""
False-alarm rate and detection rate of the split-half test, on null datasets built from the real con images.

IN   design                           stopsignal/functionnectome/tract_regression/design_<setA>_<setB>_<mask>_mean.mat
                                      made by build_regression_design.py
     contrast                         SuccStop_vs_Go, UnsuccStop_vs_Go or SuccStop_vs_UnsuccStop
     n datasets                       per condition and shape source
     workers                          optional, default 1
     project root                     optional, default $PROJECT_ROOT
     --t                              optional, compare group t-maps instead of group-mean maps
     the con images of both sets, as read by split_half_test.py

OUT  <design folder>/validate_splithalf_<setA>_<setB>_<contrast>_<mask>_mean_<n datasets>[_t].csv
"""

import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_tmap import group_t  # noqa: E402
from split_half_test import (ALPHA, N_SPLITS, SEED, beta_operator, group_mean, load_stacks, p_value,  # noqa: E402
                             read_design, split_half)

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
PROJECTION_PREFIXES = ("Striatal Bundle/External Capsule", "Muratoff Bundle", "Anterior Thalamic Radiation",
                       "Superior Thalamic Radiation", "Acoustic Radiation", "Optic Radiation",
                       "Corticospinal Tract")
CONDITIONS = ("null", "planted")
PROGRESS_STEPS = 40
OUT_NAME = "validate_splithalf_{}_{}_{}_{}_{}{}.csv"
# ====================================================================

_G = {}


def unit(v):
    v = v - v.mean()
    return v / v.std()


def run_one(job):
    cond, shapeIdx, d = job
    t0 = time.time()
    rng = np.random.default_rng(SEED + 100000 * (CONDITIONS.index(cond) * 2 + shapeIdx) + d)
    signs = rng.choice((-1.0, 1.0), size=_G["n"])[:, None]
    A = _G["patA"][cond][shapeIdx] + signs * _G["devA"]
    B = _G["patB"][shapeIdx] + signs * _G["devB"]
    res = split_half(A, B, _G["B"], N_SPLITS, SEED + d, _G["stat"])
    p, undefined = p_value(res)
    return (cond, shapeIdx, d, p, undefined, float(np.nanmedian(res[:, 4])), float(np.nanmedian(res[:, 5])),
            time.time() - t0)


def main():
    useT = "--t" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--t"]
    if len(args) not in (3, 4, 5):
        sys.exit("usage: validate_split_half.py <design .mat> <contrast> <n datasets> [workers] [project root] [--t]")
    design, contrast, nData = args[0], args[1], int(args[2])
    workers = int(args[3]) if len(args) >= 4 else 1
    root = args[4] if len(args) == 5 else DEFAULT_ROOT
    D, sets, _, maskStem = read_design(design, contrast)
    out = os.path.join(os.path.dirname(design), OUT_NAME.format(sets[0], sets[1], contrast, maskStem, nData,
                                                                "_t" if useT else ""))
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")

    X = D["X"]
    tract = [str(np.squeeze(t)).strip() for t in np.ravel(D["tract"])]
    projIdx = [j for j, t in enumerate(tract) if t.startswith(PROJECTION_PREFIXES)]
    missing = [p for p in PROJECTION_PREFIXES if not any(tract[j].startswith(p) for j in projIdx)]
    if missing:
        sys.exit(f"no tract matches: {missing}")

    stacks = load_stacks(D, sets, contrast, root)
    means = [s.mean(axis=0) for s in stacks]
    Bop = beta_operator(X)
    shapes, planted = [], []
    for m in means:
        u = unit(m)
        beta = Bop @ u
        shapes.append(u)
        planted.append(unit(u - 2 * X[:, projIdx] @ beta[projIdx]))
    patB = [means[1].mean() + means[1].std() * u for u in shapes]
    patA = {"null": [means[0].mean() + means[0].std() * u for u in shapes],
            "planted": [means[0].mean() + means[0].std() * u for u in planted]}
    _G.update(n=stacks[0].shape[0], devA=stacks[0] - means[0], devB=stacks[1] - means[1],
              patA=patA, patB=patB, B=Bop, stat=group_t if useT else group_mean)

    print(f"design {design}\ncontrast {contrast}   sets {sets[0]} / {sets[1]}   n = {_G['n']}")
    print(f"map compared per split: {'group t' if useT else 'group mean'}")
    print(f"voxels {X.shape[0]}   tracts {X.shape[1]}   datasets per condition and shape {nData}"
          f"   splits {N_SPLITS}   workers {workers}")
    print(f"projection tracts ({len(projIdx)}): {', '.join(tract[j] for j in projIdx)}")
    for k, s in enumerate(sets):
        print(f"   {s}: group-mean map over voxels mean {means[k].mean():+.4f}  SD {means[k].std():.4f}")
    for k, s in enumerate(sets):
        r = np.corrcoef(Bop @ shapes[k], Bop @ planted[k])[0, 1]
        print(f"   shape from {s}: r between null and planted tract betas {r:+.3f}")

    jobs = [(c, k, d) for c in CONDITIONS for k in range(len(sets)) for d in range(nData)]
    t0 = time.time()
    step = max(1, len(jobs) // PROGRESS_STEPS)
    results = []
    pool = Pool(workers) if workers > 1 else None
    for n, res in enumerate(pool.imap(run_one, jobs) if pool else map(run_one, jobs), 1):
        results.append(res)
        if n % step == 0 or n == len(jobs):
            print(f"  {n}/{len(jobs)} datasets, {time.time() - t0:.0f} s", flush=True)
    if pool:
        pool.close()
        pool.join()
    print(f"\n{len(jobs)} datasets in {time.time() - t0:.0f} s  "
          f"(median {np.median([r[7] for r in results]):.1f} s per dataset)")

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "shape_from", "dataset", "p", "undefined_splits", "distinguishable",
                    "median_cross", "median_reference", "seconds"])
        for cond, k, d, p, u, mc, mr, sec in results:
            w.writerow([cond, sets[k], d, f"{p:.4f}", u, int(p < ALPHA), f"{mc:.4f}", f"{mr:.4f}", f"{sec:.1f}"])
    for cond in CONDITIONS:
        for k, s in enumerate(sets):
            ps = np.array([r[3] for r in results if r[0] == cond and r[1] == k])
            print(f"   {cond:<8} shape from {s:<5} distinguishable in {int((ps < ALPHA).sum())} of {len(ps)}"
                  f" ({100 * (ps < ALPHA).mean():.1f}%)   median p {np.median(ps):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
