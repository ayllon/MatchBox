#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alejandro.alvarezayllon@unige.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

set -e

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate matchbox

set -x

MATCHDIR="${HOME}/MatchBox"
ID="dc2_g0_$(date +%Y%m%d)"
mkdir -p "${MATCHDIR}/results/$ID"

srun --exclusive -N1 -n1 "${MATCHDIR}/bin/benchmark.py" \
    --id "${ID}" \
    --repeat 1000 \
    --output-dir "${MATCHDIR}/results/" \
    --timeout 3000 \
    --no-find2 \
    --no-grow \
    --lambdas 0.05 \
    --gammas 1000 \
    -- "${MATCHDIR}/data/dc2/"*.fits

