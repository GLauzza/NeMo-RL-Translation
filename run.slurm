#!/bin/bash
#SBATCH --job-name="NEMO-RL"
#SBATCH --output=/lustre/fsn1/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --constraint=a100
#SBATCH --account=knb@a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=1:00:00

module purge
module load arch/a100
module load uv/0.8.3
module load cudnn/9.10.2.21-12-cuda
module load cuda/12.8.0
module load nccl/2.27.3-1-cuda
export TORCH_CUDA_ARCH_LIST=8.0

source /lustre/fsn1/projects/rech/knb/ukq43aj/rl/nemo-rl/.venv/bin/activate

set -x
export WANDB_MODE=offline

export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1200

NRL_FORCE_REBUILD_VENVS=true srun uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_4B_megatron.yaml
