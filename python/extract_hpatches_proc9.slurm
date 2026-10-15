#!/bin/bash -l
#SBATCH --job-name=hpatches_extract
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-%j.out

module load cuda/12.8

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python
ROOT=/srv/storage/datasets/cauamagalhaes/datasets/hpatches-release
OUT=outputs

echo "=== alike ==="
$VENV extract_easy_features.py $ROOT --elf-model alike --elf-conf confs/alike_fullres.json --output-name $OUT/alike_fullres
echo "=== aliked ==="
$VENV extract_easy_features.py $ROOT --elf-model aliked --elf-conf confs/aliked_fullres.json --output-name $OUT/aliked_fullres
echo "=== dedode (B) ==="
$VENV extract_easy_features.py $ROOT --elf-model dedode --elf-conf confs/dedodeB_fullres.json --output-name $OUT/dedodeB_fullres
echo "=== sfd2 ==="
$VENV extract_easy_features.py $ROOT --elf-model sfd2 --elf-conf confs/sfd2_fullres.json --output-name $OUT/sfd2_fullres
echo "=== disk ==="
$VENV extract_easy_features.py $ROOT --elf-model disk --elf-conf confs/disk_fullres.json --out-size 64 --output-name $OUT/disk_fullres
echo "=== superpoint ==="
$VENV extract_easy_features.py $ROOT --elf-model superpoint --elf-conf confs/superpoint_fullres.json --output-name $OUT/superpoint_fullres
echo "=== xfeat ==="
$VENV extract_easy_features.py $ROOT --elf-model xfeat --elf-conf confs/xfeat_fullres.json --output-name $OUT/xfeat_fullres
echo "=== orb ==="
$VENV extract_easy_features.py $ROOT --elf-model orb --elf-conf confs/orb_fullres.json --output-name $OUT/orb_fullres
echo "=== sift (OpenCV, native script) ==="
(cd $OUT && $VENV ../extract_opencv_sift.py $ROOT 65)

echo "=== ALL DONE ==="
