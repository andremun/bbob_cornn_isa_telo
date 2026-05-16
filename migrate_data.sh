#!/bin/bash
# =============================================================================
# migrate_data.sh
# ---------------
# Migrates existing data files from the old TELO_DATA directory layout to the
# new unified bbob_cornn_isa layout used by the refactored cornn_telo codebase.
#
# Old layout (under ~/punim0320/):
#   TELO_DATA/INPUT_data/              -> Sobol grids
#   TELO_DATA/BBOB_data/raw_data/      -> BBOB landscape evaluations
#   TELO_DATA/BBOB_data/ela_data/      -> BBOB ELA features
#   TELO_DATA/CORNN_data/raw_data/     -> CORNN landscape evaluations
#   TELO_DATA/CORNN_data/ela_data/     -> CORNN ELA features
#   BBOB_data/nevergrad_data/          -> BBOB nevergrad runs
#   BBOB_data/adam_data/               -> BBOB Adam runs
#   CORNN_data/nevergrad_data/         -> CORNN nevergrad runs (empty)
#   CORNN_data/adam_data/              -> CORNN Adam runs (empty)
#   BBOB_data/meta_raw/bbob_fopt.csv   -> BBOB optimal values
#
# New layout (under ~/punim0320/bbob_cornn_isa/):
#   input/                             <- Sobol grids
#   bbob/raw/                          <- BBOB landscape evaluations
#   bbob/ela/                          <- BBOB ELA features
#   bbob/nevergrad/                    <- BBOB nevergrad runs
#   bbob/adam/                         <- BBOB Adam runs
#   bbob/meta/                         <- BBOB metadata (fopt)
#   cornn/raw/                         <- CORNN landscape evaluations
#   cornn/ela/                         <- CORNN ELA features
#   cornn/nevergrad/                   <- CORNN nevergrad runs
#   cornn/adam/                        <- CORNN Adam runs
#   isa/                               <- MATLAB post-processing output
#
# Usage:
#   bash migrate_data.sh [--dry-run]
#
# With --dry-run, no files are copied; only the plan is printed.
# Without arguments, all files are copied (existing files are skipped).
# =============================================================================

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY RUN] No files will be copied."
fi

BASE=~/punim0320

OLD_INPUT="${BASE}/TELO_DATA/INPUT_data"
OLD_BBOB_RAW="${BASE}/TELO_DATA/BBOB_data/raw_data"
OLD_BBOB_ELA="${BASE}/TELO_DATA/BBOB_data/ela_data"
OLD_BBOB_NG="${BASE}/BBOB_data/nevergrad_data"
OLD_BBOB_ADAM="${BASE}/BBOB_data/adam_data"
OLD_BBOB_META="${BASE}/BBOB_data/meta_raw"
OLD_CORNN_RAW="${BASE}/TELO_DATA/CORNN_data/raw_data"
OLD_CORNN_ELA="${BASE}/TELO_DATA/CORNN_data/ela_data"
OLD_CORNN_NG="${BASE}/CORNN_data/nevergrad_data"
OLD_CORNN_ADAM="${BASE}/CORNN_data/adam_data"

NEW_ROOT="${BASE}/bbob_cornn_isa"
NEW_INPUT="${NEW_ROOT}/input"
NEW_BBOB_RAW="${NEW_ROOT}/bbob/raw"
NEW_BBOB_ELA="${NEW_ROOT}/bbob/ela"
NEW_BBOB_NG="${NEW_ROOT}/bbob/nevergrad"
NEW_BBOB_ADAM="${NEW_ROOT}/bbob/adam"
NEW_BBOB_META="${NEW_ROOT}/bbob/meta"
NEW_CORNN_RAW="${NEW_ROOT}/cornn/raw"
NEW_CORNN_ELA="${NEW_ROOT}/cornn/ela"
NEW_CORNN_NG="${NEW_ROOT}/cornn/nevergrad"
NEW_CORNN_ADAM="${NEW_ROOT}/cornn/adam"
NEW_ISA="${NEW_ROOT}/isa"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

make_dir() {
    if [ $DRY_RUN -eq 0 ]; then
        mkdir -p "$1"
    else
        echo "  [mkdir] $1"
    fi
}

copy_dir() {
    local src="$1"
    local dst="$2"
    if [ ! -d "$src" ]; then
        echo "  [SKIP]  Source not found: $src"
        return
    fi
    local n
    n=$(find "$src" -maxdepth 1 -type f | wc -l)
    echo "  [copy]  $src -> $dst  ($n files)"
    if [ $DRY_RUN -eq 0 ]; then
        mkdir -p "$dst"
        # rsync: skip existing files, preserve timestamps
        rsync -a --ignore-existing "$src"/ "$dst"/
    fi
}

# ---------------------------------------------------------------------------
# Create new directory tree
# ---------------------------------------------------------------------------
echo "Creating new directory layout under ${NEW_ROOT} ..."
for d in "$NEW_INPUT" "$NEW_BBOB_RAW" "$NEW_BBOB_ELA" "$NEW_BBOB_NG" \
          "$NEW_BBOB_ADAM" "$NEW_BBOB_META" "$NEW_CORNN_RAW" "$NEW_CORNN_ELA" \
          "$NEW_CORNN_NG" "$NEW_CORNN_ADAM" "$NEW_ISA"; do
    make_dir "$d"
done

# ---------------------------------------------------------------------------
# Copy data
# ---------------------------------------------------------------------------
echo ""
echo "Copying data ..."
copy_dir "$OLD_INPUT"     "$NEW_INPUT"
copy_dir "$OLD_BBOB_RAW"  "$NEW_BBOB_RAW"
copy_dir "$OLD_BBOB_ELA"  "$NEW_BBOB_ELA"
copy_dir "$OLD_BBOB_NG"   "$NEW_BBOB_NG"
copy_dir "$OLD_BBOB_ADAM" "$NEW_BBOB_ADAM"
copy_dir "$OLD_BBOB_META" "$NEW_BBOB_META"
copy_dir "$OLD_CORNN_RAW" "$NEW_CORNN_RAW"
copy_dir "$OLD_CORNN_ELA" "$NEW_CORNN_ELA"
copy_dir "$OLD_CORNN_NG"  "$NEW_CORNN_NG"
copy_dir "$OLD_CORNN_ADAM" "$NEW_CORNN_ADAM"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "File counts in new layout:"
for d in "$NEW_INPUT" "$NEW_BBOB_RAW" "$NEW_BBOB_ELA" "$NEW_BBOB_NG" \
          "$NEW_BBOB_ADAM" "$NEW_BBOB_META" "$NEW_CORNN_RAW" "$NEW_CORNN_ELA" \
          "$NEW_CORNN_NG" "$NEW_CORNN_ADAM"; do
    if [ -d "$d" ]; then
        n=$(find "$d" -maxdepth 1 -type f | wc -l)
        printf "  %6d  %s\n" "$n" "$d"
    fi
done

echo ""
if [ $DRY_RUN -eq 1 ]; then
    echo "Dry run complete. Run without --dry-run to execute."
else
    echo "Migration complete."
    echo "Old data directories have NOT been removed."
    echo "Once you have verified the new layout, you may remove them with:"
    echo "  rm -rf ${BASE}/TELO_DATA ${BASE}/BBOB_data ${BASE}/CORNN_data"
fi
