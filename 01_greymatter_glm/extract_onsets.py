#!/usr/bin/env python3
"""
Build each subject's SPM onsets file.

IN   stopsignal/data/<sub>/func/<sub>_task-stopsignal_acq-seq_events.tsv
     subjects_stopsignal_all.txt   one participant ID per line, in this folder
OUT  stopsignal/staged/<sub>/sess1/sess1.mat
"""

import os

import numpy as np
import pandas as pd
import scipy.io

# ================================ CONFIG ================================
TASK_NAME = "stopsignal"
ACQ_LABEL = "acq-seq"

DATA_ROOT = os.path.join(os.environ["PROJECT_ROOT"], "stopsignal", "data")
STAGED_ROOT = os.path.join(os.environ["PROJECT_ROOT"], "stopsignal", "staged")
SESSION_LABEL = "sess1"

SUBJECT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "subjects_stopsignal_all.txt")

GO, SUCC, UNSUCC = "go", "succesful_stop", "unsuccesful_stop"
CONDITIONS = ["go_correct", "go_error", SUCC, UNSUCC]
# ========================================================================


def load_subjects(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")]


def events_path(sub):
    return os.path.join(DATA_ROOT, sub, "func",
                        f"{sub}_task-{TASK_NAME}_{ACQ_LABEL}_events.tsv")


def condition_masks(ev):
    """Trial selection per condition. Keys must match CONDITIONS."""
    tt = ev["trial_type"].str.strip()
    acc = ev["response_accuracy"].str.strip()
    return {
        "go_correct": (tt == GO) & (acc == "correct"),
        "go_error": (tt == GO) & (acc.isin(["incorrect", "miss"])),
        SUCC: tt == SUCC,
        UNSUCC: tt == UNSUCC,
    }


def main():
    subjects = load_subjects(SUBJECT_FILE)
    print(f"subjects: {len(subjects)} (from {os.path.basename(SUBJECT_FILE)})\n")

    shapes, failed = {}, []

    for sub in subjects:
        path = events_path(sub)
        if not os.path.exists(path):
            print(f"{sub}  SKIPPED -- events file not found: {path}")
            failed.append(sub)
            continue

        ev = pd.read_csv(path, sep="\t")
        masks = condition_masks(ev)

        names, onsets, durations = [], [], []
        for cond in CONDITIONS:
            sel = ev[masks[cond]]
            if len(sel) == 0:
                continue
            names.append(cond)
            onsets.append(np.asarray(sel["onset"], dtype=float).reshape(-1, 1))
            durations.append(np.asarray(sel["duration"], dtype=float).reshape(-1, 1))

        if not names:
            print(f"{sub}  SKIPPED -- no usable trials at all")
            failed.append(sub)
            continue

        # shape required by SPM: names Nx1, onsets/durations 1xN
        mat = {
            "names": np.array([[n] for n in names], dtype=object),
            "onsets": np.array([onsets], dtype=object),
            "durations": np.array([durations], dtype=object),
        }

        out_dir = os.path.join(STAGED_ROOT, sub, SESSION_LABEL)
        os.makedirs(out_dir, exist_ok=True)
        scipy.io.savemat(os.path.join(out_dir, f"{SESSION_LABEL}.mat"), mat,
                         oned_as="column")

        counts = ", ".join(f"{n}={len(o)}" for n, o in zip(names, onsets))
        print(f"{sub}  {len(names)} conditions  [{counts}]")
        shapes.setdefault(tuple(names), []).append(sub)

    print("\n" + "=" * 72)
    print("DESIGN SHAPES WRITTEN")
    for names, subs in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(subs):3d} subjects: {' + '.join(names)}")
        if len(subs) <= 10:
            print(f"       {', '.join(subs)}")

    ok = sum(len(v) for v in shapes.values())
    print(f"\nwritten: {ok}/{len(subjects)}"
          + (f"   FAILED: {failed}" if failed else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()
