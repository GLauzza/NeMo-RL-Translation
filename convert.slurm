#!/bin/bash
#SBATCH --job-name="NEMO-RL"
#SBATCH --output=/lustre/fsn1/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/knb/ukq43aj/logs/log_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --constraint=h100
#SBATCH --account=knb@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=1:00:00

module purge
module load arch/h100
module load uv/0.8.3
module load cudnn/9.10.2.21-12-cuda
module load cuda/12.8.0
module load nccl/2.27.3-1-cuda
export TORCH_CUDA_ARCH_LIST=9.0

source /lustre/fsn1/projects/rech/knb/ukq43aj/rl/nemo-rl/.venv/bin/activate

set -x
export WANDB_MODE=offline

export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1200


srun uv run examples/converters/convert_megatron_to_hf.py --config=$SCRATCH/rl/nemo-rl/results/grpo_4b_megatron_3/step_60/config.yaml --megatron-ckpt-path=$SCRATCH/rl/nemo-rl/results/grpo_4b_megatron_3/step_60/policy/weights/iter_0000000 --hf-ckpt-path=$SCRATCH/Models/Qwen3-4B-Thinking-2507-SFT-3-nemorl-60
