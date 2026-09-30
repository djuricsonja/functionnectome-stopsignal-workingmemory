# 02_template_and_route

Correction between fMRIPrep's MNI152NLin2009cAsym and the priors' template, and each participant's input to the Functionnectome. Scripts are listed in the order they were run.

1. `register_templates.py` (`cc`)
2. `check_transform_order.py`
3. `build_all_inputs.sh`, with `build_funtome_input.py` (`lanczosWindowedSinc`)
4. `compare_funtome_inputs.py` (74 participants, inputs built by `build_funtome_input.py` with `linear`, `bSpline` and `lanczosWindowedSinc`)
