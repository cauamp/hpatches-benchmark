#!/bin/bash -l
#SBATCH --job-name=hpatches_mha
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-mha-%j.out

module load cuda/12.8

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python
ROOT=/srv/storage/datasets/cauamagalhaes/datasets/hpatches-sequences-release
RES=mha_results

echo "=== alike ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model alike --elf-conf confs/alike_fullres.json --results-dir $RES
echo "=== aliked ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model aliked --elf-conf confs/aliked_fullres.json --results-dir $RES
echo "=== dedode (B) ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model dedode --elf-conf confs/dedodeB_fullres.json --results-dir $RES --name dedodeB
echo "=== sfd2 ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model sfd2 --elf-conf confs/sfd2_fullres.json --results-dir $RES
echo "=== disk ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model disk --elf-conf confs/disk_fullres.json --results-dir $RES
echo "=== superpoint ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model superpoint --elf-conf confs/superpoint_fullres.json --results-dir $RES
echo "=== xfeat ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model xfeat --elf-conf confs/xfeat_fullres.json --results-dir $RES
echo "=== orb ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model orb --elf-conf confs/orb_fullres.json --results-dir $RES

echo "=== sift ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model sift --results-dir $RES

echo "=== ALL DONE ==="
