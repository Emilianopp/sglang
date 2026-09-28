#!/bin/bash
#SBATCH --job-name=sglang-kve-e2e
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --output=%x-%j.out
# End-to-end validation of --enable-kv-eviction: three server configurations
# (default / eager / chunked prefill) on Qwen3-0.6B plus default on Qwen3-4B,
# one GPU each, each followed by the e2e client with an exact HF reference.
set -u
REPO=${REPO:-$SCRATCH/kv-race/sglang}
OUT=${OUT:-$SCRATCH/kv-race/sglang_kve_e2e/$SLURM_JOB_ID}
mkdir -p "$OUT"
# Run from a node-local copy of the venv: concurrent cold imports of
# transformers from network scratch intermittently fail with ENOENT.
VENV=$REPO/.venv
if [ -n "${SLURM_TMPDIR:-}" ]; then
  rsync -a "$REPO/.venv/" "$SLURM_TMPDIR/venv/" && VENV=$SLURM_TMPDIR/venv
  sed -i "s#$REPO/.venv#$VENV#g" "$VENV/bin/activate"
fi
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# The cu13 wheels ship a full toolkit (nvcc, headers); JIT users (deep_ep,
# flashinfer) need CUDA_HOME, and Mila modules stop at CUDA 12.6.
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export CUDA_HOME=$SITE/nvidia/cu13 PATH=$SITE/nvidia/cu13/bin:$PATH
cd "$OUT"   # neutral cwd (repo-root cwd confuses platform detection)
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee gpus.csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())" | tee torch.txt

run_cfg() {  # gpu port tag model extra-args...
  local gpu=$1 port=$2 tag=$3 model=$4; shift 4
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python -m sglang.launch_server --model-path "$model" --port "$port" \
      --enable-streaming-session --enable-kv-eviction --page-size 1 \
      --mem-fraction-static 0.55 "$@" > "server_$tag.log" 2>&1 &
    spid=$!
    for i in $(seq 1 360); do
      curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
      kill -0 $spid 2>/dev/null || { echo "[$tag] server died"; exit 2; }
      sleep 5
    done
    python "$REPO/test/manual/kv_eviction/e2e_kv_eviction.py" \
      --url "http://127.0.0.1:$port" --model "$model" --tag "$tag" \
      > "client_$tag.log" 2>&1
    rc=$?
    echo "[$tag] client rc=$rc"
    kill $spid; wait $spid 2>/dev/null
    exit $rc
  ) &
}

# Staggered: simultaneous cold imports from network scratch have failed reads.
# Staggered starts keep JIT/compile caches from racing. CONFIGS selects a
# subset, e.g. CONFIGS="default default4b".
CONFIGS=${CONFIGS:-"default eager chunked default4b"}
for c in $CONFIGS; do
  case $c in
    default)   run_cfg 0 31000 default Qwen/Qwen3-0.6B ;;
    eager)     run_cfg 1 31001 eager Qwen/Qwen3-0.6B --disable-cuda-graph --disable-overlap-schedule ;;
    chunked)   run_cfg 2 31002 chunked Qwen/Qwen3-0.6B --chunked-prefill-size 64 ;;
    default4b) run_cfg 3 31003 default4b Qwen/Qwen3-4B ;;
  esac
  sleep 45
done
wait
grep -h "RESULT\|FAIL" client_*.log
