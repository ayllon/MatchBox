#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alejandro.alvarezayllon@unige.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00

set -e

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate matchbox

set -x

MATCHDIR="${HOME}/MatchBox"
ID="chembl_$(date +%Y%m%d)"
mkdir -p "${MATCHDIR}/results/$ID"

for i in {82..160..3}; do
    "${MATCHDIR}/bin/benchmark.py" --id "${ID}" \
        --output-dir "${MATCHDIR}/results/" \
        --lambdas 0.05 \
        --bootstrap-alpha 0.05 \
        --no-find2 \
        --repeat 10 \
        --files $i \
        --timeout 3000 \
        ${MATCHDIR}/data/chembl/chembl_??.db
done

