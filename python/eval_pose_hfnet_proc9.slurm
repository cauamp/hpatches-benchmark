#!/bin/bash -l
#SBATCH --job-name=pose1500_hfnet
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-pose-hfnet-%j.out

module load singularity-ce/4.3.2

HFNET_DIR=/home/phmgc/projects/TPAMI/xfeat-tpami25-src/matching_benchmarks/hfnet
IMAGE=/srv/storage/singularity/images/guipotje/hfnet_standalone.sif
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python

cd /tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python

echo "=== starting hfnet server ==="
# --nv: TF's GPU-enabled build needs libcuda.so.1 present at import time even
# though hfnet_server.py forces CPU-only compute (CUDA_VISIBLE_DEVICES=-1);
# no GPU gres is requested since it's never used for actual computation.
singularity exec --nv \
  --bind "${HFNET_DIR}/hfnet_client.py":/app/hfnet_client.py,"${HFNET_DIR}/hfnet_server.py":/app/hfnet_server.py \
  "${IMAGE}" \
  python3 /app/hfnet_server.py &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT

# wait for the server to be listening before the client connects
for i in $(seq 1 30); do
  if (echo > /dev/tcp/localhost/5005) 2>/dev/null; then
    echo "server ready after ${i}s"
    break
  fi
  sleep 1
done

for ds in megadepth scannet; do
  echo "=== hfnet / $ds ==="
  $VENV eval_pose.py --dataset $ds --elf-model hfnet --results-dir pose_results --n-workers $SLURM_CPUS_PER_TASK
done

echo "=== ALL DONE ==="
