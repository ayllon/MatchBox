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
ID="ailerons_start3_$(date +%Y%m%d)"
mkdir -p "${MATCHDIR}/results/$ID"

for i in $(seq 1 $SLURM_NTASKS); do
    srun --exclusive -N1 -n1 "${MATCHDIR}/bin/benchmark.py" \
        --id "${ID}" \
        --bootstrap-arity 3 \
        --uind-alpha 0.1 \
        --bootstrap-alpha 0.1 \
        --lambdas 0.00 0.05 0.10 \
        --repeat 20 \
        --timeout 3000 \
        --output-dir "${MATCHDIR}/results/" \
        "${MATCHDIR}/data/keel/ailerons/ailerons.dat" "${MATCHDIR}/data/keel/elevators/elevators.dat" &> "${MATCHDIR}/results/$ID/run.${i}.log" &
done

wait

