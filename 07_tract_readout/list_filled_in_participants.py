"""
Participants whose filled-in signal share exceeds a threshold in any tract of any of the given files.

IN   threshold                        fraction, 0.01 in the analysis
     out txt                          filled_in_above1pc_proj_asso.txt in the analysis; not overwritten if it exists
     share csv ...                    logs/missing_signal_share_by_tract_<task>_<set>_<contrast>_<wm stem>.csv, one or more
                                      (proj and asso), made by measure_missing_signal_share_by_tract.py

OUT  out txt, one participant per line, sorted
"""

import csv
import os
import sys


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: list_filled_in_participants.py <threshold, fraction> <out .txt> <share .csv> [...]")
    threshold, out, sources = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
    if os.path.exists(out):
        sys.exit(f"output exists, not overwriting: {out}")
    union = set()
    for src in sources:
        with open(src, newline="") as fh:
            over = sorted({r["subject"] for r in csv.DictReader(fh) if float(r["share"]) > threshold})
        print(f"{os.path.basename(src)}: {len(over)} above {threshold}: {' '.join(over)}")
        union |= set(over)
    with open(out, "w", newline="\n") as fh:
        fh.writelines(s + "\n" for s in sorted(union))
    print(f"union: {len(union)}: {' '.join(sorted(union))}\nwrote {out}")


if __name__ == "__main__":
    main()
