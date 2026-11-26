#!/bin/bash
#SBATCH --job-name=clinAI_61_sofa_nestedcv
#SBATCH --output="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/61_sofa_nested_%j.out"
#SBATCH --error="/home/tcmhwd/microbiome/ICU_public/clinical metadata/log/61_sofa_nested_%j.err"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu_partition

# Nested-CV rebuild of the SOFA-referenced model family (S0/S1/S2a/S2b/
# S3ref/S3a/S3b), per
# semantic_microbiome/reports/paper1/script41_nested_rebuild_DESIGN.md
# (deep-reasoner, 2026-09-05). Does NOT modify script 41/45/46 or
# utils/nested_meta.py in place -- runs the new nested-CV sibling script
# only, strictly additive (new result files, nothing overwritten). See that
# script's own docstring for the full bug/fix description, the five
# reproduction gates, and the two deliberate departures from script 41.
#
# Cost driver: 90 base-model fits (3 layers x 30 = 5 outer x (5 inner + 1
# outer-full)), the SAME 90 fits script 45 ran in ~10 min at this exact
# allocation -- NOT the bootstrap (paired bootstrap/NRI/DCA together are
# under a minute in aggregate; every bootstrap resample scores 302 ALREADY-
# COMPUTED probabilities, no refits). See the design doc's cost table.
#
# --cpus-per-task=4, exactly matching script 45/57's allocation:
#   (i) measured -- LogisticRegressionCV's inner fits go 0.7s -> 0.5s from
#       1->4 cores and flat beyond, on matrices of this size.
#   (ii) reproducibility -- the strict per-patient <1e-6 reproduction gate
#        (Gates A/B/C below) has only ever been demonstrated at 4 cores
#        (script 57 against script 46). Raising the allocation buys nothing
#        and risks tripping a gate for a reason unrelated to the science.
# The BLAS thread-limit exports below are the prerequisite that makes this
# allocation behave on a 256-core cpu_partition node (otherwise OpenBLAS/MKL
# spawn one thread per NODE core for 302x929 matrix ops -- pure contention).
#
# They are set to 1, NOT to $SLURM_CPUS_PER_TASK. Job 587411 hit the previous
# 45-minute limit still inside the base-probability block, having done in
# >44 min what script 45 does in ~10. The exports were =4 there, and the
# parallelism inside a layer is already LogisticRegressionCV(n_jobs=4): four
# joblib workers each permitted four BLAS threads is 16 threads on a 4-core
# cgroup. joblib normally caps worker BLAS threads at 1 for exactly this
# reason, but honours an explicitly exported value instead. Script 57 set
# =4 and was unaffected only because M1 (16 markers, 6-7 s) is too small for
# BLAS threading to matter; script 61 is the first to run the M3-LR layer
# (430 s in script 45) under those exports.
#
# =1 also matches the reference environment more closely than =4 did:
# script 45's wrapper exports nothing, so its workers ran at joblib's
# default of one inner thread -- and script 45/46's persisted vectors are
# what the <1e-6 reproduction gates compare against. Script 60 separately
# showed the base fits reproduce to 4 dp between 1 and 4 threads.
#
# Node contention is not excluded as a contributing cause (node2 was at
# 218/256 allocated CPUs), which is why the script now logs per-layer wall
# times against script 45's 7 s / 430 s / 58 s reference.
#
# --mem=8G: peak working set is a 302x929 float matrix plus a few dozen
# full-length probability vectors -- tens of megabytes; matches script 45.
# --time=03:00:00: raised from 00:45:00 after the timeout above. Generous
# rather than tuned, deliberately -- the per-layer timings from this run are
# what a future estimate should be based on, and a second timeout would buy
# no information at all.
# No --nodelist: nothing here needs a specific node.
#
# On any of the five reproduction gates failing (see script docstring):
# the script aborts BEFORE opening any output file and writes nothing. If
# Gate B fails only on the CatBoost (C2b) layer, the one permitted retry is
# resubmitting at this SAME --cpus-per-task=4 / BLAS-thread configuration
# (already what this wrapper sets); if it still fails, do not relax the
# tolerance -- report the magnitude to the owner.
#
# Usage:
#   sbatch "/home/tcmhwd/microbiome/ICU_public/clinical metadata/slurm/61_phase_x_microbiome_over_sofa_nested.sh"

set -eo pipefail
source /data/tcmhwd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate gpt
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE="/home/tcmhwd/microbiome/ICU_public/clinical metadata"
mkdir -p "$BASE/log"
cd "$BASE/scripts"

echo "=== host: $(hostname); cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}; OMP/OPENBLAS/MKL_NUM_THREADS: $OMP_NUM_THREADS/$OPENBLAS_NUM_THREADS/$MKL_NUM_THREADS ==="

python3 61_phase_x_microbiome_over_sofa_nested.py

echo "61_phase_x_microbiome_over_sofa_nested.sh complete."
