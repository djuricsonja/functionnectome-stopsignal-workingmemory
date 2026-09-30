#!/bin/bash
# Build every participant's Functionnectome input, in parallel.
#
# IN   subjects_stopsignal_all.txt   one participant ID per line, in this folder
#      build_funtome_input.py
#
# OUT  stopsignal/functionnectome/input/<sub>.nii.gz
#      stopsignal/functionnectome/input/<sub>_support.nii.gz
#      logs/build_input/<sub>.log

set -u

python -c "import ants, nibabel" || { echo "ERROR: venv not usable"; exit 1; }
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
OUT="$PROJECT_ROOT/stopsignal/functionnectome/input"
LOGS="$PROJECT_ROOT/logs/build_input"
SUBJECTS="$SCRIPTS/subjects_stopsignal_all.txt"
INTERPOLATOR="lanczosWindowedSinc"
WORKERS=36

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

mkdir -p "$OUT" "$LOGS"
total=$(grep -c . "$SUBJECTS")
echo "interpolator $INTERPOLATOR, $WORKERS workers, $total subjects"
df -h "$OUT" | tail -1
echo

n=0
while read -r s; do
    [ -z "$s" ] && continue
    n=$((n + 1))
    if [ -f "$OUT/$s.nii.gz" ] && [ -f "$OUT/${s}_support.nii.gz" ]; then
        echo "[$n/$total] $s already built, skipping"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 2; done
    echo "[$n/$total] $s started"
    python -u "$SCRIPTS/build_funtome_input.py" "$s" "$INTERPOLATOR" "$OUT" \
        > "$LOGS/$s.log" 2>&1 &
done < "$SUBJECTS"
wait

echo
echo "=== verification ==="
missing=0
while read -r s; do
    [ -z "$s" ] && continue
    if [ ! -f "$OUT/$s.nii.gz" ] || [ ! -f "$OUT/${s}_support.nii.gz" ]; then
        echo "  MISSING: $s"
        missing=$((missing + 1))
    fi
done < "$SUBJECTS"
echo "subjects without both outputs: $missing"
grep -h "RESULT nonzero outside the support" "$LOGS"/*.log \
    | awk '{s+=$6; if($6>m) m=$6} END {print "leak outside support: total "s", worst "m}'
grep -lE "ERROR|Traceback" "$LOGS"/*.log | head
du -sh "$OUT"
df -h "$OUT" | tail -1
echo "ALL DONE"
