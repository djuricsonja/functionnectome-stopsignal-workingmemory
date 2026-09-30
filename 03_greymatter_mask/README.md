# 03_greymatter_mask

The group grey-matter mask, built from each participant's anatomical parcellation, and what each priors set draws from it. Scripts are listed in the order they were run.

1. `build_group_gm_mask.py`
2. `write_group_gm_mask.py`
3. `build_group_aseg_counts.py` (`priors`, then `arm1`)
4. `write_group_aseg_labels.py` (`priors`, then `arm1`)
5. `report_aseg_agreement_curve.py` (`priors`), with `FreeSurferSubcorticalLabelTableLut.txt` from HCP Pipelines v6.0.0
6. `write_group_gm_mask_aseg.py`, with the same label table
7. `report_tissue_mask_composition.py`
8. `compare_masks_by_structure.py` (the 50% label map, the tissue-based mask, the anatomical mask)
9. `measure_priors_source_by_structure.py` (`proj`, `asso`, `wb`, `comm`, with the anatomical mask and the 50% label map)
10. `report_combined_source_share.py` (the `proj` output of step 9)
11. `measure_priors_lost_weight.py` (`proj`, `asso`, `wb`, `comm`, with the tissue-based mask, then the anatomical mask)
