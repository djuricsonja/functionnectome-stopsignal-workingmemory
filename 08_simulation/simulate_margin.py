"""
Sensitivity of the tract-profile tests of H1 and H2, on artificial second tasks built from the real con images.

IN   mode                             h1, h2 or h2power
     design                           stopsignal/functionnectome/tract_regression/design_<setA>_<setB>_<mask>_mean.mat
                                      made by build_regression_design.py (07_tract_readout): wb wb for h1, proj asso for h2 and h2power
     contrast                         SuccStop_vs_Go, UnsuccStop_vs_Go or SuccStop_vs_UnsuccStop
     n datasets                       per level
     workers                          optional, default 1
     project root                     optional, default $PROJECT_ROOT
     the con images of the design's sets, as read by split_half_test.py (07_tract_readout)

OUT  <design folder>/simulate_margin_<mode>_<contrast>_<mask>_mean_<n datasets>.csv   one row per dataset
"""

import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_half_test import (ALPHA, N_SPLITS, beta_operator, group_mean, load_stacks, p_value,  # noqa: E402
                             read_design, split_half)

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
SEED = 20260928
MODES = ("h1", "h2", "h2power")
LEVELS = {"h1": (1.00, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85, 0.80),
          "h2": (1.00, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85, 0.80),
          "h2power": tuple((round(hi - gap, 2), hi) for hi in (1.00, 0.95)
                           for gap in (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20))}
H1_SET = "wb"
H2_LOWER, H2_HIGHER = "proj", "asso"
H2_SUPPORT = 0.95
GUARD_TOL = 1e-6
EQUIV_TOL = 1e-9
PROGRESS_STEPS = 40
OUT_NAME = "simulate_margin_{}_{}_{}_{}.csv"
# ====================================================================

_G = {}


def turn_profile(b, rho, rng):
    """Betas correlating with b at exactly rho, with b's mean and centred length."""
    mu = b.mean()
    u = b - mu
    z = rng.permutation(b) - mu
    z = z - (z @ u) / (u @ u) * u
    nz = np.linalg.norm(z)
    if nz < 1e-12 * np.linalg.norm(u):
        raise ValueError("shuffle parallel to the profile")
    z *= np.linalg.norm(u) / nz
    return mu + rho * u + np.sqrt(max(0.0, 1.0 - rho * rho)) * z


def plant(m, X, B, rho, rng):
    """Group-mean map m with its tract betas turned to correlate at rho with the original; guarded."""
    b = B @ m
    b2 = turn_profile(b, rho, rng)
    m2 = m + X @ (b2 - b)
    got = B @ m2
    r = np.corrcoef(got, b)[0, 1]
    if abs(r - rho) > GUARD_TOL or np.abs(got - b2).max() > GUARD_TOL * max(1.0, np.abs(b2).max()):
        raise ValueError(f"planted profile missed: target r {rho}, recovered r {r:.8f}")
    return m2


def halves(n, n_rounds, seed, resample):
    """Index pairs per round: split_half's random halves, optionally resampled with replacement within each half."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rounds):
        perm = rng.permutation(n)
        h1, h2 = perm[: n // 2], perm[n // 2:]
        if resample:
            h1, h2 = rng.choice(h1, size=len(h1)), rng.choice(h2, size=len(h2))
        out.append((h1, h2))
    return out


def split_half_betas(PA, PB, n_splits, seed, idx=None):
    """split_half on per-participant tract betas (participants x tracts); same generator use, same columns.
    idx, if given, replaces the random halves."""
    idx = halves(PA.shape[0], n_splits, seed, False) if idx is None else idx
    out = np.empty((len(idx), 6))
    for s, (h1, h2) in enumerate(idx):
        bA1, bA2 = PA[h1].mean(axis=0), PA[h2].mean(axis=0)
        bB1, bB2 = PB[h1].mean(axis=0), PB[h2].mean(axis=0)
        rAA = np.corrcoef(bA1, bA2)[0, 1]
        rBB = np.corrcoef(bB1, bB2)[0, 1]
        rA1B2 = np.corrcoef(bA1, bB2)[0, 1]
        rA2B1 = np.corrcoef(bA2, bB1)[0, 1]
        prod = rAA * rBB
        out[s] = (rAA, rBB, rA1B2, rA2B1, (rA1B2 + rA2B1) / 2, np.sqrt(prod) if prod >= 0 else np.nan)
    return out


def h2_share(resLow, resHigh):
    """Share of rounds, among those with both references above zero, where the lower set's corrected similarity is
    below the higher set's; and the number of rounds left out."""
    ok = (resLow[:, 5] > 0) & (resHigh[:, 5] > 0)
    cLow, cHigh = resLow[ok, 4] / resLow[ok, 5], resHigh[ok, 4] / resHigh[ok, 5]
    return (float((cLow < cHigh).mean()) if ok.any() else float("nan")), int((~ok).sum())


def make_tasks(k, rho, rng, s1, s2):
    """Per-participant betas of task 1 and task 2 for set k, and task 2's noise-free map."""
    m, dev_b = _G["means"][k], _G["dev_b"][k]
    m2 = plant(m, _G["X"], _G["B"], rho, rng)
    b1, b2 = _G["B"] @ m, _G["B"] @ m2
    return b1 + s1[:, None] * dev_b, b2 + s2[:, None] * dev_b, m2


def run_one(job):
    li, d = job
    t0 = time.time()
    rho = _G["levels"][li]
    rng = np.random.default_rng([SEED, li, d])
    s1 = rng.choice((-1.0, 1.0), size=_G["n"])
    s2 = rng.choice((-1.0, 1.0), size=_G["n"])
    splitSeed = int(rng.integers(2 ** 32))
    idx = halves(_G["n"], N_SPLITS, splitSeed, _G["mode"] != "h1")
    rhos = rho if isinstance(rho, tuple) else (rho,) * len(_G["sets"])
    res, maps = [], []
    for k in range(len(_G["sets"])):
        PA, PB, m2 = make_tasks(k, rhos[k], rng, s1, s2)
        res.append(split_half_betas(PA, PB, N_SPLITS, splitSeed, idx))
        maps.append(m2)
    if _G["mode"] == "h1":
        result, undefined = p_value(res[0])
        verdict = int(result < ALPHA)
    else:
        result, undefined = h2_share(res[0], res[1])
        verdict = int(result >= H2_SUPPORT)
    med = [float(np.nanmedian(r[:, c])) for r in res for c in (4, 5)]
    extra = (s1, s2, splitSeed, maps) if (li, d) == (0, 0) else None
    return li, d, result, verdict, undefined, med, time.time() - t0, extra


def fmt_level(level):
    """'0.95', or '0.90|0.95' for a (lower set, higher set) pair."""
    return "|".join(f"{v:.2f}" for v in level) if isinstance(level, tuple) else f"{level:.2f}"


def check_equivalence(extra):
    """First dataset recomputed on voxel maps with split_half_test.split_half."""
    s1, s2, splitSeed, maps = extra
    worst = 0.0
    for k in range(len(_G["sets"])):
        A = _G["means"][k] + s1[:, None] * _G["dev"][k]
        C = maps[k] + s2[:, None] * _G["dev"][k]
        ref = split_half(A, C, _G["B"], N_SPLITS, splitSeed, group_mean)
        m = _G["means"][k]
        PA = _G["B"] @ m + s1[:, None] * _G["dev_b"][k]
        PB = _G["B"] @ maps[k] + s2[:, None] * _G["dev_b"][k]
        fast = split_half_betas(PA, PB, N_SPLITS, splitSeed)
        worst = max(worst, float(np.nanmax(np.abs(ref - fast))))
    return worst


def main():
    if len(sys.argv) not in (5, 6, 7) or sys.argv[1] not in MODES:
        sys.exit("usage: simulate_margin.py <h1|h2|h2power> <design .mat> <contrast> <n datasets> [workers]"
                 " [project root]")
    mode, design, contrast, nData = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) >= 6 else 1
    root = sys.argv[6] if len(sys.argv) == 7 else DEFAULT_ROOT
    D, sets, contrasts, maskStem = read_design(design, contrast)
    want = [H1_SET] if mode == "h1" else [H2_LOWER, H2_HIGHER]
    missing = [s for s in want if s not in sets]
    if missing:
        sys.exit(f"sets {missing} not in the design's sets {sets}")
    out = os.path.join(os.path.dirname(design), OUT_NAME.format(mode, contrast, maskStem, nData))
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")

    X = D["X"]
    stacks = load_stacks(D, sets, contrast, root)
    b = contrasts.index(contrast)
    print(f"design {design}\nmode {mode}   contrast {contrast}   sets used {', '.join(want)}")
    for a, s in enumerate(sets):
        diff = np.abs(group_mean(stacks[a]) - D["Y"][:, a, b]).max()
        print(f"full-sample mean vs design ({s}): max abs difference {diff:.3e}")
    idx = [sets.index(s) for s in want]
    Bop = beta_operator(X)
    means = [stacks[i].mean(axis=0) for i in idx]
    dev = [stacks[i] - means[j] for j, i in enumerate(idx)]
    _G.update(mode=mode, sets=want, levels=LEVELS[mode], n=stacks[0].shape[0], X=X, B=Bop, means=means, dev=dev,
              dev_b=[d_ @ Bop.T for d_ in dev])
    del stacks
    n = _G["n"]
    print(f"n = {n}   halves {n // 2} / {n - n // 2}   voxels {X.shape[0]}   tracts {X.shape[1]}")
    print(f"levels {', '.join(fmt_level(v) for v in _G['levels'])}   datasets per level {nData}   splits {N_SPLITS}"
          f"   workers {workers}   seed {SEED}")

    jobs = [(li, d) for li in range(len(_G["levels"])) for d in range(nData)]
    t0 = time.time()
    step = max(1, len(jobs) // PROGRESS_STEPS)
    results = []
    pool = Pool(workers) if workers > 1 else None
    for k, r in enumerate(pool.imap(run_one, jobs) if pool else map(run_one, jobs), 1):
        if r[7] is not None:
            worst = check_equivalence(r[7])
            print(f"first dataset, betas vs voxel maps: max abs difference {worst:.3e}", flush=True)
            if not worst <= EQUIV_TOL:
                if pool:
                    pool.terminate()
                sys.exit("split on betas differs from split_half on voxel maps; stopped")
        results.append(r[:7])
        if k % step == 0 or k == len(jobs):
            print(f"  {k}/{len(jobs)} datasets, {time.time() - t0:.0f} s", flush=True)
    if pool:
        pool.close()
        pool.join()

    medNames = [f"median_{q}_{s}" for s in want for q in ("cross", "reference")]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "dataset", "p" if mode == "h1" else "share_lower_below_higher",
                    "called_different" if mode == "h1" else "h2_supported", "undefined_splits"] + medNames
                   + ["seconds"])
        for li, d, res, verdict, und, med, sec in results:
            w.writerow([fmt_level(_G["levels"][li]), d, f"{res:.4f}", verdict, und] + [f"{v:.4f}" for v in med]
                       + [f"{sec:.2f}"])
    label = "called different" if mode == "h1" else "H2 supported"
    print(f"\n{len(jobs)} datasets in {time.time() - t0:.0f} s")
    for li, rho in enumerate(_G["levels"]):
        v = np.array([r[3] for r in results if r[0] == li], dtype=float)
        rate = v.mean()
        half = 1.96 * np.sqrt(rate * (1 - rate) / len(v))
        und = sum(r[4] for r in results if r[0] == li)
        print(f"   true r {fmt_level(rho)}   {label} in {int(v.sum())} of {len(v)} ({100 * rate:.1f}%, "
              f"95% interval {100 * max(0, rate - half):.1f}-{100 * min(1, rate + half):.1f}%)   "
              f"undefined splits {und}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
