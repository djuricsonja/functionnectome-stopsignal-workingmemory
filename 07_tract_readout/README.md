# 07_tract_readout

The readout of the projected group maps by tract, and the tests between the priors sets. Scripts are listed in the order they were run.

1. `write_group_wm_mask.py`
2. `measure_regression_voxels.py` (`proj`, `asso`, the white-matter mask)
3. `build_regression_design.py` (`proj` and `asso`, then `wb` and `wb`, with the white-matter mask)
4. `tract_regression.m` (the `proj`/`asso` design)
5. `validate_split_half.py` (the `proj`/`asso` design, `SuccStop_vs_UnsuccStop`, 500 datasets)
6. `split_half_test.py` (the `proj`/`asso` design, each contrast)
7. `paired_tract_test.py` (the `proj`/`asso` design, each contrast)
8. `measure_missing_signal_share_by_tract.py` (`stopsignal`, `proj` and `asso`, the anatomical and white-matter masks)
9. `list_filled_in_participants.py` (`0.01`, the `proj` and `asso` outputs of step 8)
10. `one_sample_tract_test.py` (each contrast, then each contrast with the list of step 9 excluded)

`group_tmap.py` holds functions that steps 5, 6 and 7 use.
