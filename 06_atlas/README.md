# 06_atlas

The XTRACT tract atlas on the priors' grid, and the tract each of its volumes holds. Scripts are listed in the order they were run.

1. `resample_atlas_flirt.sh` (XTRACT release 2.0.0 at 1 mm)
2. `check_resampled_atlas.py` (the 2 mm output, the 1 mm original)
3. `compare_xtract_releases.py` (release 1.2.0 against release 2.0.0)
4. `check_xtract_labels.py` (release 1.2.0, then release 2.0.0, with the XTRACT protocols)
5. `build_xtract_volume_table.py` (the matches of step 3, the release 2.0.0 label file), which writes `xtract_2.0.0_volume_table.csv`
