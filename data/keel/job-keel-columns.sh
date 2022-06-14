#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alejandro.alvarezayllon@unige.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

set -e

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate matchbox

set -x

MATCHDIR="${HOME}/MatchBox"
ID="incremental_$(date +%Y%m%d)"
mkdir -p "${MATCHDIR}/results/$ID"

for i in {29..31}; do
    "${MATCHDIR}/bin/benchmark.py" --id "${ID}" \
        --output-dir "${MATCHDIR}/results/" \
        --lambdas 0.05 \
        --bootstrap-alpha 0.1 \
        --no-find2 \
        --repeat 10 \
        --files $i \
        --timeout 3000 \
        ${MATCHDIR}/data/keel/*/*.dat
done

