#!/bin/bash -l
#SBATCH --job-name=hpatches_mha_zippy
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-mha-zippy-%j.out

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/.venv-zippypoint/bin/python
ROOT=/srv/storage/datasets/cauamagalhaes/datasets/hpatches-sequences-release

echo "=== zippypoint ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model zippypoint --results-dir mha_results

echo "=== ALL DONE ==="
