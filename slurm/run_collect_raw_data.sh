#!/bin/bash
# ===========================================================================
# run_collect_raw_data.sh  —  Stage 1: collect raw landscape data
#
# Task-space layout (1404 total tasks, 2 per job):
#   Segment 1: BBOB raw    tasks    1 – 1080  (24 fns x 15 inst x 3 dims)
#   Segment 2: CORNN raw   tasks 1081 – 1404  (54 fns x 6 archs)
#
# Submit first; then submit run_collect_pflacco.sh with:
#   sbatch --dependency=afterok:<JOBID> run_collect_pflacco.sh
# ===========================================================================
#SBATCH --nodes=1
#SBATCH --job-name="collect_raw"
#SBATCH --ntasks=1
#SBATCH --partition=sapphire
#SBATCH --mem=64GB
#SBATCH --array=1-1000
#SBATCH --time=7-0:0:00

if [ "x$SLURM_JOB_ID" == "x" ]; then
    echo "You need to submit your job to the queuing system with sbatch"
    exit 1
fi

BBOB_RAW_START=1;    BBOB_RAW_END=1080
CORNN_RAW_START=1081; CORNN_RAW_END=1404

TOTAL_TASKS=1404
ARRAY_SIZE=1000
TASKS_PER_JOB=$(( (TOTAL_TASKS + ARRAY_SIZE - 1) / ARRAY_SIZE ))

module purge
module load foss/2022a tqdm/4.64.0 scikit-learn/1.1.2 PyTorch/1.12.1-CUDA-11.7.0

source ~/venvs/CORNN/bin/activate
cd ~/venvs/CORNN/CORNN/

for k in $(seq 1 $TASKS_PER_JOB); do
    FLAT_TASK=$(( (SLURM_ARRAY_TASK_ID - 1) * TASKS_PER_JOB + k ))
    [ $FLAT_TASK -gt $TOTAL_TASKS ] && break
    echo "Array ${SLURM_ARRAY_TASK_ID}, sub-task ${k}: flat ${FLAT_TASK}"

    if [ $FLAT_TASK -le $BBOB_RAW_END ]; then
        LOCAL_IDX=$(( FLAT_TASK - BBOB_RAW_START + 1 ))
        echo "  -> BBOB raw, local ${LOCAL_IDX}"
        TASK_ID=$LOCAL_IDX python bbob_collect_raw_data.py

    elif [ $FLAT_TASK -le $CORNN_RAW_END ]; then
        LOCAL_IDX=$(( FLAT_TASK - CORNN_RAW_START + 1 ))
        echo "  -> CORNN raw, local ${LOCAL_IDX}"
        TASK_ID=$LOCAL_IDX python cornn_collect_raw_data.py
    fi
done

deactivate
##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
