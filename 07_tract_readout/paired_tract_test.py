"""
Paired tract test: each participant's contrast map regressed on the tracts in both sets, the
difference between the sets per tract tested across participants.

IN   design                           stopsignal/functionnectome/tract_regression/design_<setA>_<setB>_<mask>_mean.mat
                                      made by build_regression_design.py
     contrast                         SuccStop_vs_Go, UnsuccStop_vs_Go or SuccStop_vs_UnsuccStop
     project root                     optional, default $PROJECT_ROOT
     the con images of both sets, as read by split_half_test.py

OUT  <design folder>/per_participant_<setA>_<setB>_<contrast>_<mask>_mean.csv         one row per tract
     <design folder>/per_participant_betas_<setA>_<setB>_<contrast>_<mask>_mean.csv   one row per participant and tract
"""

import csv
import os
import sys

import numpy as np
from scipy.stats import false_discovery_control, ttest_rel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_tmap import FUNTOME_DIR, SUBJECT_LIST, read_lines  # noqa: E402
from split_half_test import ALPHA, beta_operator, load_stacks, read_design  # noqa: E402

# ============================== CONFIG ==============================
DEFAULT_ROOT = os.environ["PROJECT_ROOT"]
OUT_TEST = "per_participant_{}_{}_{}_{}.csv"
OUT_BETAS = "per_participant_betas_{}_{}_{}_{}.csv"
# ====================================================================


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        sys.exit("usage: paired_tract_test.py <design .mat> <contrast> [project root]")
    design, contrast = args[0], args[1]
    root = args[2] if len(args) == 3 else DEFAULT_ROOT
    D, sets, _, stem = read_design(design, contrast)
    folder = os.path.dirname(design)
    outTest = os.path.join(folder, OUT_TEST.format(sets[0], sets[1], contrast, stem))
    outBetas = os.path.join(folder, OUT_BETAS.format(sets[0], sets[1], contrast, stem))
    for p in (outTest, outBetas):
        if os.path.exists(p):
            sys.exit(f"output exists, not overwriting: {p}")

    X = D["X"]
    tract = [str(np.squeeze(t)).strip() for t in np.ravel(D["tract"])]
    subjects = sorted(read_lines(os.path.join(root, FUNTOME_DIR, sets[0], "group", contrast, SUBJECT_LIST)))
    A, B = load_stacks(D, sets, contrast, root)
    Bop = beta_operator(X)
    bA, bB = A @ Bop.T, B @ Bop.T                          # participants x tracts
    t, p = ttest_rel(bA, bB, axis=0)
    q = false_discovery_control(p, method="bh")
    diff = bA - bB
    n = diff.shape[0]
    print(f"design {design}\ncontrast {contrast}   sets {sets[0]} / {sets[1]}   n = {n}   tracts {len(tract)}")

    rows = []
    for j, name in enumerate(tract):
        rows.append({"tract": name, f"mean_beta_{sets[0]}": bA[:, j].mean(), f"mean_beta_{sets[1]}": bB[:, j].mean(),
                     "mean_difference": diff[:, j].mean(), "sd_difference": diff[:, j].std(ddof=1),
                     "t": t[j], "df": n - 1, "p": p[j], "q_fdr": q[j], "significant": int(q[j] < ALPHA)})
    with open(outTest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(outBetas, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "tract", f"beta_{sets[0]}", f"beta_{sets[1]}"])
        for i, s in enumerate(subjects):
            for j, name in enumerate(tract):
                w.writerow([s, name, f"{bA[i, j]:.6g}", f"{bB[i, j]:.6g}"])

    sig = [r for r in rows if r["significant"]]
    print(f"tracts with q < {ALPHA}: {len(sig)} of {len(tract)}")
    for r in sorted(sig, key=lambda r: r["q_fdr"]):
        print(f"   {r['tract']:<46} {sets[0]} - {sets[1]} {r['mean_difference']:+.4f}   t({n - 1}) {r['t']:+.2f}"
              f"   q {r['q_fdr']:.2e}")
    print(f"wrote {outTest}\nwrote {outBetas}")


if __name__ == "__main__":
    main()
