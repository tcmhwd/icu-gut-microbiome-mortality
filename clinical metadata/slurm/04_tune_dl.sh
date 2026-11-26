#!/bin/bash
#SBATCH --job-name=clinAI_04_tune_dl
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/04_tune_dl_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/04_tune_dl_%j.err"
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=cpu_partition
#SBATCH --nodelist=node2

# Step 4/5 -- Optuna hyperparameter tuning of MLP, FT-Transformer, and TabNet
# (20 trials/family/direction by default -- see below; 2 directions x 3 families).
#
# CPU-ONLY (2026-08-12): GPU (gpu_partition/node1) is temporarily unavailable
# on this cluster, so this submits to cpu_partition/node2 by default. No code
# changes were needed for CPU fallback -- utils/dl_models.py already falls
# back to CPU automatically (torch.cuda.is_available() check), and
# 05_tune_dl.py caps torch's thread pool to $SLURM_CPUS_PER_TASK so it
# doesn't oversubscribe the shared node.
#
# To switch back to GPU once it's available again, change:
#   --partition=cpu_partition  -> --partition=gpu_partition
#   --nodelist=node2           -> --nodelist=node1
# and add back: #SBATCH --gres=gpu:1
#
# RECOMMENDED USAGE -- submit per family, not all 3 in one job (2026-08-12
# operational finding): on CPU, per-epoch cost differs enormously by family
# (observed: MLP ~1 sec/epoch, FT-Transformer ~50 sec/epoch due to attention,
# TabNet ~13-26 min/fold). Running all 3 sequentially in one job means MLP
# finishes in hours while the job as a whole is gated by the slowest family
# for days. Submit one job per (family, dev dataset) instead, e.g.:
#
#   for DEV in ICU MIMIC; do
#     sbatch --job-name=clinAI_04_tune_dl_MLP_$DEV \
#       --output="log/04_tune_dl_MLP_${DEV}_%j.out" --error="log/04_tune_dl_MLP_${DEV}_%j.err" \
#       --cpus-per-task=8  slurm/04_tune_dl.sh --families MLP           --dev-datasets $DEV
#     sbatch --job-name=clinAI_04_tune_dl_FTTransformer_$DEV \
#       --output="log/04_tune_dl_FTTransformer_${DEV}_%j.out" --error="log/04_tune_dl_FTTransformer_${DEV}_%j.err" \
#       --cpus-per-task=12 slurm/04_tune_dl.sh --families FTTransformer --dev-datasets $DEV
#     sbatch --job-name=clinAI_04_tune_dl_TabNet_$DEV \
#       --output="log/04_tune_dl_TabNet_${DEV}_%j.out" --error="log/04_tune_dl_TabNet_${DEV}_%j.err" \
#       --cpus-per-task=8  slurm/04_tune_dl.sh --families TabNet        --dev-datasets $DEV
#   done
#
# `--dev-datasets` (default: both) restricts a job to one direction, letting
# each (family, direction) run as its own concurrent job. Spreading these
# across node1 (CPU-idle even with GPU busy -- submit with
# --partition=gpu_partition --nodelist=node1, no --gres needed) and node2
# gives real parallelism. Keep total requested CPUs comfortably under each
# node's PHYSICAL core count (node1: 64, node2: 128 -- see `sinfo -N -o
# "%N %C %z"`, format is alloc/idle/other/total and socket:core:thread) --
# SLURM allocates logical (hyperthread) CPUs, and requesting more logical
# CPUs than physical cores causes real contention/slowdown for CPU-bound
# jobs sharing the node (observed 2026-08-12: over-allocating slowed down an
# unrelated PyCaret screening job on the same node by >7x).
#
# KNOWN ISSUE: pytorch-tabnet occasionally SIGSEGVs at startup under SLURM's
# cgroup-constrained CPU affinity (observed 2026-08-12; not reproduced
# locally without cgroup CPU pinning -- looks like a race in native
# (MKL/OpenMP) thread-pool init, not a code bug here). In practice it has
# only been observed on node1, repeatedly (2/2 attempts), never on node2
# (0/3+ attempts) -- if a TabNet job's .err shows "Segmentation fault",
# prefer resubmitting it on node2 (--partition=cpu_partition --nodelist=node2)
# over retrying on node1.
#
# Requires slurm/01_preprocess.sh to have completed.
#
# Studies are resumable (sqlite storage under clinical_ai/optuna/DL/), same
# as slurm/03_tune_ml.sh. Trained models are saved as single loadable
# bundles under clinical_ai/trained_models/DL/<dev>/<model>/ -- see
# scripts/utils/model_io.py (load_bundle() / list_available_models()).
#
# Usage: sbatch "slurm/04_tune_dl.sh"                          # all 3 families, both directions, one job
# Common overrides:
#   sbatch "slurm/04_tune_dl.sh" --families MLP TabNet
#   sbatch "slurm/04_tune_dl.sh" --dev-datasets ICU
#   sbatch "slurm/04_tune_dl.sh" --n-trials 10 --max-epochs 60

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u   # the 'gpt' env's conda activate.d hooks (MKL) are not `set -u`-safe
conda activate gpt
set -u

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE"

python3 "$BASE/scripts/05_tune_dl.py" "$@"
