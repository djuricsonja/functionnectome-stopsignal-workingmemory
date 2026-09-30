# Paths for every script in this repository. Fill in the three paths, then run once per session:
#   source set_paths.sh
# FSLDIR and FREESURFER_HOME are set by FSL's and FreeSurfer's own setup.

export PROJECT_ROOT=/path/to/project
export SPM_PATH=/path/to/spm12
export COLLDIAG_PATH=/path/to/colldiag

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDERS="$REPO/01_greymatter_glm:$REPO/02_template_and_route:$REPO/03_greymatter_mask:$REPO/04_coverage:$REPO/05_projection:$REPO/06_atlas:$REPO/07_tract_readout:$REPO/08_simulation"
export PYTHONPATH="$FOLDERS${PYTHONPATH:+:$PYTHONPATH}"
export MATLABPATH="$FOLDERS${MATLABPATH:+:$MATLABPATH}"
