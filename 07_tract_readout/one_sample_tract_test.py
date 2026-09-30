"""
One-sample tract test: per set and tract, each participant's tract beta tested against zero across participants.

IN   betas csv                        <design folder>/per_participant_betas_<setA>_<setB>_<contrast>_<mask>_mean.csv
                                      made by paired_tract_test.py
     --exclude <list>                 optional, one participant per line, made by list_filled_in_participants.py

OUT  <design folder>/one_sample_<setA>_<setB>_<contrast>_<mask>_mean[_excl-<list stem>].csv   one row per set and tract
"""

import csv
import os
import sys

import numpy as np
from scipy.stats import false_discovery_control, ttest_1samp

# ============================== CONFIG ==============================
ALPHA = 0.05
IN_PREFIX = "per_participant_betas_"
OUT_PREFIX = "one_sample_"
EXCL_SUFFIX = "_excl-{}"
# ====================================================================


def main():
    args = sys.argv[1:]
    if len(args) not in (1, 3) or (len(args) == 3 and args[1] != "--exclude"):
        sys.exit("usage: one_sample_tract_test.py <per_participant_betas .csv> [--exclude <exclusion .txt>]")
    src = args[0]
    name = os.path.basename(src)
    if not name.startswith(IN_PREFIX):
        sys.exit(f"input name does not start with {IN_PREFIX}: {name}")
    stem, ext = os.path.splitext(name[len(IN_PREFIX):])
    exclude = set()
    if len(args) == 3:
        with open(args[2]) as fh:
            exclude = {line.strip() for line in fh if line.strip()}
        if not exclude:
            sys.exit(f"exclusion list is empty: {args[2]}")
        stem += EXCL_SUFFIX.format(os.path.splitext(os.path.basename(args[2]))[0])
    out = os.path.join(os.path.dirname(src), OUT_PREFIX + stem + ext)
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")

    with open(src, newline="") as fh:
        reader = csv.DictReader(fh)
        sets = [c[len("beta_"):] for c in reader.fieldnames if c.startswith("beta_")]
        rows = list(reader)
    present = {r["subject"] for r in rows}
    subjects = sorted(present - exclude)
    tracts = list(dict.fromkeys(r["tract"] for r in rows))
    index = {(r["subject"], r["tract"]): r for r in rows}
    missing = [(s, t) for s in subjects for t in tracts if (s, t) not in index]
    if missing:
        sys.exit(f"{len(missing)} participant-tract pairs missing, first {missing[0]}")
    print(f"input {src}")
    if exclude:
        left = sorted(exclude & present)
        absent = sorted(exclude - present)
        print(f"excluded {len(left)} of {len(exclude)} listed: {' '.join(left)}")
        if absent:
            print(f"listed but not in the input: {' '.join(absent)}")
    print(f"sets {' / '.join(sets)}   n = {len(subjects)}   tracts {len(tracts)}")

    results = []
    for s in sets:
        b = np.array([[float(index[(sub, t)][f"beta_{s}"]) for t in tracts] for sub in subjects])
        t, p = ttest_1samp(b, 0.0, axis=0)
        q = false_discovery_control(p, method="bh")
        n = b.shape[0]
        setRows = [{"set": s, "tract": tracts[j], "n": n, "mean_beta": b[:, j].mean(), "sd_beta": b[:, j].std(ddof=1),
                    "t": t[j], "df": n - 1, "p": p[j], "q_fdr": q[j], "significant": int(q[j] < ALPHA)}
                   for j in range(len(tracts))]
        results += setRows
        sig = sorted((r for r in setRows if r["significant"]), key=lambda r: r["q_fdr"])
        print(f"{s}: tracts with q < {ALPHA}: {len(sig)} of {len(tracts)}")
        for r in sig:
            print(f"   {r['tract']:<46} mean beta {r['mean_beta']:+.4f}   t({n - 1}) {r['t']:+.2f}   q {r['q_fdr']:.2e}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
