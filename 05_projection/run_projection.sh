#!/bin/bash
# Project every participant in a list through one priors set with the Functionnectome.
#
# IN   label                            output folder name under stopsignal/functionnectome/
#      priors file                      file name in priors/, e.g. priors_proj_proba_3T.h5
#      priors label                     the toolbox's name for that priors set
#      mask                             stopsignal/masks/group_aseg/group_gm_mask.nii.gz, made by write_group_gm_mask_aseg.py (03_greymatter_mask)
#      subject list                     optional, default subjects_stopsignal_all.txt, one participant ID per line, in this folder
#      n processes                      optional, default 24
#      n subjects                       optional, first n participants only
#      stopsignal/functionnectome/input/<sub>.nii.gz   made by build_all_inputs.sh (02_template_and_route)
#
# OUT  stopsignal/functionnectome/<label>/voxelwise_analysis/<sub>/functionnectome.nii.gz
#      stopsignal/functionnectome/<label>.fcntm
#
# RESUME=1 continues an interrupted run with the same mask.

set -u

python -c "import ants, nibabel" || { echo "ERROR: venv not usable"; exit 1; }

FUNTOME="$PROJECT_ROOT/stopsignal/functionnectome"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_SUBJECTS="$SCRIPTS/subjects_stopsignal_all.txt"
DEFAULT_NPROC=24

if [ "$#" -lt 4 ]; then
    echo "Usage: run_projection.sh <label> <priors file> <priors label> <mask>" \
         "[<subject list>] [<n processes>] [<n subjects>]"
    exit 1
fi
LABEL="$1"
PRIORS="$PROJECT_ROOT/priors/$2"
PRIORS_LABEL="$3"
MASK="$4"
SUBJECTS="${5:-$DEFAULT_SUBJECTS}"
NPROC="${6:-$DEFAULT_NPROC}"
LIMIT="${7:-0}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

for f in "$PRIORS" "$MASK" "$SUBJECTS"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done

out="$FUNTOME/$LABEL"
settings="$FUNTOME/$LABEL.fcntm"

existing=$(find "$out/voxelwise_analysis" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$existing" -gt 0 ] && [ "${RESUME:-0}" != "1" ]; then
    echo "ERROR: $out/voxelwise_analysis already holds $existing subject folders."
    echo "The cached normalisation sum belongs to the mask that wrote them, so reusing this"
    echo "folder with another mask divides every voxel by the wrong denominator."
    echo "Remove the folder for a fresh run, or set RESUME=1 to continue one with the same mask."
    exit 1
fi
mkdir -p "$out"

list=$(grep . "$SUBJECTS")
if [ "$LIMIT" -gt 0 ]; then
    list=$(echo "$list" | head -n "$LIMIT")
fi
nsub=$(echo "$list" | grep -c .)

missing=0
for s in $list; do
    [ -f "$FUNTOME/input/$s.nii.gz" ] || { echo "MISSING INPUT: $s"; missing=$((missing + 1)); }
done
[ "$missing" -eq 0 ] || { echo "ERROR: $missing inputs missing"; exit 1; }

{
    echo "Output folder:"
    echo "$out"
    echo "Analysis ('voxel' or 'region'):"
    echo "voxel"
    echo "Number of parallel processes:"
    echo "$NPROC"
    echo "Priors stored as ('h5' or 'nii'):"
    echo "h5"
    echo "HDF5 priors:"
    echo "$PRIORS_LABEL"
    echo "Position of the subjects ID in their path:"
    echo "-1"
    echo "Mask the output:"
    echo "1"
    echo "Number of subjects:"
    echo "$nsub"
    echo "Number of masks:"
    echo "1"
    echo "Subject's BOLD paths:"
    for s in $list; do echo "$FUNTOME/input/$s.nii.gz"; done
    echo ""
    echo "Masks for voxelwise analysis:"
    echo "$MASK"
    echo ""
    echo "###"
    echo "HDF5 path:"
    echo "$PRIORS"
    echo "Template path:"
    echo ""
    echo "Probability maps (voxel) path:"
    echo ""
    echo "Probability maps (region) path:"
    echo ""
    echo "Region masks path:"
    echo ""
    echo "###"
} > "$settings"

echo "label $LABEL, priors $(basename "$PRIORS"), $nsub subjects, $NPROC processes"
echo "mask     $MASK"
echo "list     $SUBJECTS"
echo "settings $settings"
python -c "
from Functionnectome.functionnectome import readSettings
import pprint, sys
v = readSettings('$settings')
pprint.pprint({k: (str(x)[:70] + '...' if isinstance(x, list) and len(x) > 2 else x)
               for k, x in v.items()})
" || { echo "ERROR: the toolbox rejected the settings file"; exit 1; }
df -h "$out" | tail -1
echo

time Functionnectome "$settings"
echo "exit code $?"
du -sh "$out"
df -h "$out" | tail -1
echo "PROJECTION DONE"
