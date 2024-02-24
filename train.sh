#!/bin/bash
#SBATCH --job-name=fine_tune_sparql
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition= alldlc_gpu-rtx2080
#SBATCH --output /work/dlclarge2/khaterm-nltoSPARQL/finetune/logs/%x-%A-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error /work/dlclarge2/khaterm-nltoSPARQL/finetune/logs/%x-%A-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out
#SBATCH --mem 40GB
#SBATCH --time=10:00



echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
#source ~/miniconda3/bin/activate # Adjust to your path of Miniconda installation
#conda activate hello_cluster_env
source /work/dlclarge2/khaterm-nltoSPARQL/venv/bin/activate
pip install -r requirements.txt


echo "START TIME: $(date)"

# Training setup

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE= 4
echo "world size: $WORLD_SIZE"



CMD=" \
    finetune_script.py \    "

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --num_processes $WORLD_SIZE \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
"


SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1 | tee ~/logs/%x-%A-HelloCluster.out

echo "END TIME: $(date)"