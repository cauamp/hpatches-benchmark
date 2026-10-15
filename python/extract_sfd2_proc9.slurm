#!/bin/bash -l
#SBATCH --job-name=hpatches_sfd2
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-sfd2-%j.out

module load cuda/12.8

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python
ROOT=/srv/storage/datasets/cauamagalhaes/datasets/hpatches-release
OUT=outputs

echo "=== sfd2 (zero-keypoint fallback fixed) ==="
$VENV extract_easy_features.py $ROOT --elf-model sfd2 --elf-conf confs/sfd2_fullres.json --output-name $OUT/sfd2_fullres

echo "=== ALL DONE ==="
