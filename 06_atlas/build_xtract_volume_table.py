"""
Build the volume-to-tract table of a newer XTRACT release from its comparison with an older one.

IN   matches csv                      compare_1.2.0_2.0.0_matches.csv, made by compare_xtract_releases.py
     new atlas xml                    data/atlases/XTRACT.xml from FSL's fsl-data_atlases_xtract 2.0.0
     table csv                        atlases/xtract_2.0.0/xtract_2.0.0_volume_table.csv (a copy is in this folder); not overwritten if it exists

OUT  table csv, one row per volume: volume, tract, XML label, assignment, r and Dice with the older release
"""

import csv
import os
import re
import sys

# ============================== CONFIG ==============================
SOURCE_CONFIRMED = "label confirmed: best match of the same tract in the older release"
SOURCE_CORRECTED = "label corrected: best match of this tract in the older release; XML says {}"
SOURCE_NEW = "tract absent from the older release; XML label kept"
# ====================================================================


def read_labels(xmlPath):
    text = open(xmlPath, encoding="utf-8", errors="replace").read()
    pairs = re.findall(r'<label[^>]*index="(\d+)"[^>]*>([^<]+)</label>', text)
    if [int(i) for i, _ in pairs] != list(range(len(pairs))):
        sys.exit(f"label indices in {xmlPath} are not 0..n-1 in order")
    return [name.strip() for _, name in pairs]


def main():
    if len(sys.argv) != 4:
        sys.exit("give the matches csv, the newer release's XML and the output table")
    matchesPath, xmlPath, outPath = sys.argv[1:4]
    for p in (matchesPath, xmlPath):
        if not os.path.isfile(p):
            sys.exit(f"missing: {p}")
    if os.path.exists(outPath):
        sys.exit(f"output exists, not overwritten: {outPath}")

    labels = read_labels(xmlPath)
    with open(matchesPath, newline="") as fh:
        matches = list(csv.DictReader(fh))
    oldTracts = {m["old_map"] for m in matches}

    assigned = {}
    for m in matches:
        v = int(m["best_new_volume"])
        if v in assigned:
            sys.exit(f"volume {v} is the best match of both {assigned[v][0]} and {m['old_map']}")
        if labels[v] != m["best_new_label"]:
            sys.exit(f"volume {v}: matches file says {m['best_new_label']}, XML says {labels[v]}")
        source = SOURCE_CONFIRMED if m["status"] == "same" else SOURCE_CORRECTED.format(labels[v])
        assigned[v] = (m["old_map"], source, m["r_best"], m["dice_best"])

    for v, lab in enumerate(labels):
        if v in assigned:
            continue
        if lab in oldTracts:
            sys.exit(f"volume {v} is labelled {lab}, a tract of the older release, but no older tract "
                     f"matches it best")
        assigned[v] = (lab, SOURCE_NEW, "", "")

    tracts = [assigned[v][0] for v in range(len(labels))]
    dup = sorted({t for t in tracts if tracts.count(t) > 1})
    if dup:
        sys.exit(f"tracts naming more than one volume: {dup}")

    with open(outPath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["volume", "tract", "xml_label", "assignment", "r_with_older_release",
                    "dice_with_older_release"])
        for v in range(len(labels)):
            t, s, r, d = assigned[v]
            w.writerow([v, t, labels[v], s, r, d])

    nConf = sum(assigned[v][1] == SOURCE_CONFIRMED for v in assigned)
    nNew = sum(assigned[v][1] == SOURCE_NEW for v in assigned)
    nCorr = len(labels) - nConf - nNew
    print(f"volumes: {len(labels)}  confirmed {nConf}  corrected {nCorr}  new {nNew}")
    for v in range(len(labels)):
        if assigned[v][1] not in (SOURCE_CONFIRMED, SOURCE_NEW):
            print(f"  volume {v:>2}: XML {labels[v]:<40} -> {assigned[v][0]}")
    print(f"wrote {outPath}")


if __name__ == "__main__":
    main()
