# 01_greymatter_glm

Grey-matter analysis of the stop-signal task. Scripts are listed in the order they were run.

1. `extract_onsets.py`
2. `extract_motion_parameters.py`
3. `firstlevel_greymatter.m`, with `makeSPMdesignmatrix_mov_reg.m`
4. `make_contrast_stop_signal.m` (`arm1`), with `analysis_paths.m`
5. `group_level_stop_signal.m` (`arm1`)
6. `export_group_tables.m` (`arm1`)
7. `recovery_rae_coordinates.py`, with `rae_2015_table1.csv`
8. `report_response_by_structure.py` (`stopsignal/group_all`, `aseg_labels_50pc.nii.gz` from `03_greymatter_mask`, `4.8988658676`)
