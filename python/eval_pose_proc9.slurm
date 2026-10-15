#!/bin/bash -l
#SBATCH --job-name=pose1500
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-pose-%j.out
#
# Usage: sbatch eval_pose_proc9.slurm <results_dir> <model[:conf[:name]]> ...
# Set PY=<interpreter> to override the default venv (e.g. .venv-zippypoint).

module load cuda/12.8

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python
PY=${PY:-/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python}
RES=$1; shift

for spec in "$@"; do
  IFS=: read -r model conf name <<< "$spec"
  for ds in megadepth scannet; do
    echo "=== ${name:-$model} / $ds ==="
    $PY eval_pose.py --dataset $ds --elf-model $model ${conf:+--elf-conf $conf} ${name:+--name $name} \
      --results-dir $RES --n-workers $SLURM_CPUS_PER_TASK
  done
done

echo "=== ALL DONE ==="
