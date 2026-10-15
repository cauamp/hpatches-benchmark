#!/bin/bash -l
#SBATCH --job-name=hpatches_mha_hfnet
#SBATCH --partition=gpu
#SBATCH --nodelist=proc9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/tank/teams/image_matching_team/cauamp/proc9draft/hpatches-benchmark/python/logs/slurm-mha-hfnet-%j.out

module load singularity-ce/4.3.2

HFNET_DIR=/home/phmgc/projects/TPAMI/xfeat-tpami25-src/matching_benchmarks/hfnet
IMAGE=/srv/storage/singularity/images/guipotje/hfnet_standalone.sif
VENV=/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/.venv/bin/python
ROOT=/srv/storage/datasets/cauamagalhaes/datasets/hpatches-sequences-release

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

echo "=== hfnet (MHA eval) ==="
$VENV eval_hpatches_sequences.py $ROOT --elf-model hfnet --results-dir mha_results

echo "=== ALL DONE ==="
