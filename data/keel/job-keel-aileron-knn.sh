#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alejandro.alvarezayllon@unige.ch
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

set -e

eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
conda activate matchbox

set -x

MATCHDIR="${HOME}/MatchBox"
ID="ailerons_$(date +%Y%m%d)_knn"
mkdir -p "${MATCHDIR}/results/$ID"

for i in $(seq 1 $SLURM_NTASKS); do
    srun --exclusive -N1 -n1 "${MATCHDIR}/bin/benchmark.py" \
        --id "${ID}" \
        --repeat 1000 \
        --timeout 3000 \
        --sample-size 1000 \
        --no-find2 \
        --test-method knn \
        --lambdas 0.1 --gammas 1 \
        --uind-alpha 0.05 \
        --bootstrap-alpha 0.05 \
        --output-dir "${MATCHDIR}/results/" \
        "${MATCHDIR}/data/keel/ailerons/ailerons.dat" "${MATCHDIR}/data/keel/elevators/elevators.dat" &> "${MATCHDIR}/results/$ID/run.${i}.log" &
done

wait

