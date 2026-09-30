#!/usr/bin/env python3
"""
Extract the 12 motion regressors for SPM.

IN   stopsignal/data/derivatives/fmriprep/<sub>/func/<sub>_task-stopsignal_acq-seq_desc-confounds_regressors.tsv
     subjects_stopsignal_all.txt   one participant ID per line, in this folder
OUT  stopsignal/staged/<sub>/sess1/motion_sess1.txt   (12 cols, no header)
"""

import os

import pandas as pd

# ================================ CONFIG ================================
TASK_NAME = "stopsignal"
ACQ_LABEL = "acq-seq"

DATA_ROOT = os.path.join(os.environ["PROJECT_ROOT"], "stopsignal", "data")
FMRIPREP = os.path.join(DATA_ROOT, "derivatives", "fmriprep")
STAGED_ROOT = os.path.join(os.environ["PROJECT_ROOT"], "stopsignal", "staged")
SESSION_LABEL = "sess1"

SUBJECT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "subjects_stopsignal_all.txt")

# order must match the motion names in firstlevel_greymatter.m
MOTION_COLUMNS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
]
# ========================================================================


def load_subjects(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")]


def confounds_path(sub):
    return os.path.join(FMRIPREP, sub, "func",
                        f"{sub}_task-{TASK_NAME}_{ACQ_LABEL}_desc-confounds_regressors.tsv")


def main():
    subjects = load_subjects(SUBJECT_FILE)
    print(f"subjects: {len(subjects)} (from {os.path.basename(SUBJECT_FILE)})\n")

    written, failed = 0, []

    for sub in subjects:
        path = confounds_path(sub)
        if not os.path.exists(path):
            print(f"{sub}  SKIPPED -- confounds file not found: {path}")
            failed.append(sub)
            continue

        df = pd.read_csv(path, sep="\t")

        missing = [c for c in MOTION_COLUMNS if c not in df.columns]
        if missing:
            print(f"{sub}  SKIPPED -- missing columns: {missing}")
            failed.append(sub)
            continue

        motion = df[MOTION_COLUMNS]

        n_na = int(motion.isna().sum().sum())
        n_na_row1 = int(motion.iloc[0].isna().sum())
        note = ""
        if n_na != n_na_row1:
            note = f"   WARNING: {n_na} n/a cells, only {n_na_row1} in row 1"

        out_dir = os.path.join(STAGED_ROOT, sub, SESSION_LABEL)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"motion_{SESSION_LABEL}.txt")

        motion.to_csv(out_file, sep=" ", header=False, index=False, na_rep="0.0")

        print(f"{sub}  {len(motion)} rows x {len(MOTION_COLUMNS)} cols"
              f"  (n/a filled: {n_na}){note}")
        written += 1

    print("\n" + "=" * 72)
    print(f"written: {written}/{len(subjects)}"
          + (f"   FAILED: {failed}" if failed else ""))
    print("=" * 72)


if __name__ == "__main__":
    main()
