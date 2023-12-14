#!/bin/bash
# Begin LSF Directives
#SBATCH -A nph159
#SBATCH -t 06:00:00
#SBATCH -J proton
#SBATCH -o proton_run.%J
#SBATCH -e proton_run.%J
#SBATCH -N 160
#SBATCH -n 1280
#SBATCH --exclusive
#SBATCH --gpu-bind=map_gpu:0,1,2,3,7,6,5,4
##SBATCH -q debug
#export BIND="--cpu-bind=verbose,map_ldom:3,3,1,1,2,2,0,0"

module load PrgEnv-gnu craype-accel-amd-gfx90a amd-mixed rocm cray-python cray-mpich
export MPICH_GPU_SUPPORT_ENABLED=1
source /ccs/home/dbollweg/gpt/lib/cgpt/build/source.sh
export PYTHONPATH=${PYTHONPATH}:/ccs/home/dbollweg/TMD_work/gpt_utils:/ccs/home/dbollweg/TMD_work/gpt_utils/qTMD:/ccs/home/dbollweg/TMD_work/gpt_utils/utils

export OMP_NUM_THREADS=16


offset=0

for conf_num in 1690 1730 1750 1770 1790 
do
    srun -N 32 -n 256 -r $offset --gpus-per-task=1 -u python3 proton_prop.py --mpi 2.4.4.8 --mpi_split 2.4.4.4 --grid 64.64.64.128 --config_num $conf_num --shm-mpi 1 --shm 2048 --comms-sequential --accelerator-threads 8 > proton_"$conf_num".out &
    offset=$((offset + 32))
    sleep 2
done
wait
