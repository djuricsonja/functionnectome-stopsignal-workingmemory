#!/bin/bash
# Resize a 1 mm standard-space atlas to 2 mm isotropic with FSL flirt, nearest neighbour.
#
# IN   in.nii.gz                        atlas at 1 mm, 3D or 4D
#                                       (atlases/xtract_2.0.0/xtract-tract-atlases-prob-1mm.nii.gz, FSL's XTRACT atlas release 2.0.0)
# OUT  out.nii.gz                       not overwritten if it exists
#                                       (atlases/xtract_2.0.0/xtract-tract-atlases-prob-2mm_flirt_nn.nii.gz)

set -euo pipefail

# ============================== CONFIG ==============================
RESOLUTION_MM=2
INTERP=nearestneighbour
# ====================================================================

export FSLOUTPUTTYPE=NIFTI_GZ

if [ "$#" -ne 2 ]; then
    echo "give the input atlas and the output file" >&2
    exit 1
fi
IN="$1"
OUT="$2"
[ -f "$IN" ] || { echo "ERROR: not found: $IN" >&2; exit 1; }
[ -e "$OUT" ] && { echo "ERROR: output exists, not overwritten: $OUT" >&2; exit 1; }

echo "flirt: $FSLDIR/bin/flirt"
echo "in:    $IN"
echo "out:   $OUT"
echo "resolution ${RESOLUTION_MM} mm, interpolation ${INTERP}"
"$FSLDIR/bin/flirt" -in "$IN" -ref "$IN" -out "$OUT" -applyisoxfm "$RESOLUTION_MM" -interp "$INTERP"
echo "exit $?"
ls -la "$OUT"
