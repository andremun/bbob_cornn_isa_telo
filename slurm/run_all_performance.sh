#!/bin/bash
# ===========================================================================
# run_all_performance.sh  —  Full performance data collection (all algorithms)
#
# Task-space layout (7020 total tasks, 8 per job):
#   Segment 1: BBOB nevergrad  tasks    1 – 4320  (4 algs x 24 fns x 15 inst x 3 dims)
#   Segment 2: BBOB Adam       tasks 4321 – 5400  (1 alg  x 24 fns x 15 inst x 3 dims)
#   Segment 3: CORNN nevergrad tasks 5401 – 6696  (4 algs x 54 fns x 6 archs)
#   Segment 4: CORNN Adam      tasks 6697 – 7020  (1 alg  x 54 fns x 6 archs)
# ===========================================================================
#SBATCH --nodes=1
#SBATCH --job-name="CORNN_BBOB_perf"
#SBATCH --ntasks=1
#SBATCH --partition=sapphire
#SBATCH --mem=64GB
#SBATCH --array=1-1000
#SBATCH --time=7-0:0:00

if [ "x$SLURM_JOB_ID" == "x" ]; then
    echo "You need to submit your job to the queuing system with sbatch"
    exit 1
fi

BBOB_NG_START=1;     BBOB_NG_END=4320
BBOB_AD_START=4321;  BBOB_AD_END=5400
CORNN_NG_START=5401; CORNN_NG_END=6696
CORNN_AD_START=6697; CORNN_AD_END=7020

TOTAL_TASKS=7020
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

    if [ $FLAT_TASK -le $BBOB_NG_END ]; then
        LOCAL_IDX=$(( FLAT_TASK - BBOB_NG_START + 1 ))
        echo "  -> BBOB nevergrad, local ${LOCAL_IDX}"
        TASK_ID=$LOCAL_IDX python bbob_run_nevergrad.py

    elif [ $FLAT_TASK -le $BBOB_AD_END ]; then
        LOCAL_IDX=$(( FLAT_TASK - BBOB_AD_START + 1 ))
        echo "  -> BBOB Adam, local ${LOCAL_IDX}"
        TASK_ID=$LOCAL_IDX python bbob_run_adam.py

    elif [ $FLAT_TASK -le $CORNN_NG_END ]; then
        LOCAL_IDX=$(( FLAT_TASK - CORNN_NG_START + 1 ))
        echo "  -> CORNN nevergrad, local ${LOCAL_IDX}"
        TASK_ID=$LOCAL_IDX python cornn_run_nevergrad.py

    elif [ $FLAT_TASK -le $CORNN_AD_END ]; then
        LOCAL_IDX=$(( FLAT_TASK - CORNN_AD_START + 1 ))
        echo "  -> CORNN Adam, local ${LOCAL_IDX}"
        TASK_ID=$LOCAL_IDX python cornn_run_adam.py
    fi
done

deactivate
##DO NOT ADD/EDIT BEYOND THIS LINE##
my-job-stats -a -n -s
