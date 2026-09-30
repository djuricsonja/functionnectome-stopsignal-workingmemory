# 04_coverage

How much of the grey-matter mask each participant's scan did not sample, and how much of their projected signal that affects. Scripts are listed in the order they were run.

1. `report_missing_voxels.py` (the anatomical mask)
2. `report_missing_structures.py` (the anatomical mask, the 50% label map)
3. `measure_missing_signal_share.py` (`proj`, `asso`, `wb`, `comm`, with the anatomical mask)
