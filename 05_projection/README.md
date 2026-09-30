# 05_projection

Functionnectome projection of each participant's stop-signal run through the four priors sets, and the models on the projected data. Scripts are listed in the order they were run.

1. `run_projection.sh` (`proj`, `asso`, `wb`, `comm`, each with its priors file, its toolbox label such as `V2.P.Proj - Projection, Probabilistic`, and the anatomical mask)
2. `firstlevel_funtome.m` (`proj`, `asso`, `wb`, `comm`), with `makeSPMdesignmatrix_mov_reg.m` from `01_greymatter_glm`
3. `make_contrast_stop_signal.m`, `group_level_stop_signal.m` and `export_group_tables.m` from `01_greymatter_glm` (`proj`, `asso`, `wb`, `comm`)
