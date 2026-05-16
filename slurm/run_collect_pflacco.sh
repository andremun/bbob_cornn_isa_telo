#!/bin/bash
# ===========================================================================
# run_collect_pflacco.sh  —  Stage 2: compute ELA/pflacco features
#
# Submit with:  sbatch --dependency=afterok:<RAW_JOBID> run_collect_pflacco.sh
#
# Task-space layout (684 total tasks, 1 per job):
#   Segment 1: BBOB pflacco    tasks   1 –  360  (3 dims x 5 reps x 24 fns)
#   Segment 2: CORNN pflacco   tasks 361 –  684  (54 fns x 6 archs)
# ===========================================================================
#SBATCH --nodes=1
#SBATCH --job-name="collect_pflacco"
#SBATCH --ntasks=1
#SBATCH --partition=sapphire
#SBATCH --mem=64GB
#SBATCH --array=1-684
#SBATCH --time=7-0:0:00

if [ "x$SLURM_JOB_ID" == "x" ]; then
    echo "You need to submit your job to the queuing system with sbatch"
    exit 1
fi

BBOB_PF_START=1;    BBOB_PF_END=360
CORNN_PF_START=361; CORNN_PF_END=684

FLAT_TASK=$SLURM_ARRAY_TASK_ID
echo "Array ${SLURM_ARRAY_TASK_ID}: flat ${FLAT_TASK}"

module purge
module load foss/2022a tqdm/4.64.0 scikit-learn/1.1.2 PyTorch/1.12.1-CUDA-11.7.0

source ~/venvs/CORNN/bin/activate
cd ~/venvs/CORNN/CORNN/

if [ $FLAT_TASK -le $BBOB_PF_END ]; then
    LOCAL_IDX=$(( FLAT_TASK - BBOB_PF_START + 1 ))
    echo "  -> BBOB pflacco, local ${LOCAL_IDX}"
    TASK_ID=$LOCAL_IDX python bbob_run_pflacco.py

elif [ $FLAT_TASK -le $CORNN_PF_END ]; then
    LOCAL_IDX=$(( FLAT_TASK - CORNN_PF_START + 1 ))
    echo "  -> CORNN pflacco, local ${LOCAL_IDX}"
    TASK_ID=$LOCAL_IDX python cornn_run_pflacco.py
fi

deactivate
##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
