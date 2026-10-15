#!/bin/bash -l
#SBATCH --job-name=hpatches_mha_extra
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-mha-extra-%j.out

module load cuda/12.8

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python
ROOT=/srv/storage/datasets/cauamagalhaes/datasets/hpatches-sequences-release
RES=mha_results_extra_sparse

# Note: 'deal' was tested and excluded -- its easy_local_features wrapper
# produces near-random descriptors (1 mutual match out of 840 keypoints on a
# smoke-test pair), a pre-existing bug in that baseline, not this wiring.

echo "=== d2net ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model d2net --elf-conf confs/d2net_fullres.json --results-dir $RES
echo "=== delf ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model delf --results-dir $RES
echo "=== desc_reasoning ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model desc_reasoning --elf-conf confs/desc_reasoning_fullres.json --results-dir $RES
echo "=== r2d2 ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model r2d2 --elf-conf confs/r2d2_fullres.json --results-dir $RES
echo "=== superpoint_open ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model superpoint_open --elf-conf confs/superpoint_open_fullres.json --results-dir $RES

echo "=== ALL DONE ==="
