#!/bin/bash

#SBATCH --time=70:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --partition=m12
#SBATCH --mem-per-cpu=3900M
#SBATCH -J "NodeChk"
#SBATCH --mail-user=waseemashraf1584@gmail.com
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=1

module purge
module load gcc/13.2.0-hlknow5 intel-oneapi-mkl/2024.1.0-j45wg6u openmpi/5.0.3-zduzrzb python

echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

mpirun --map-by core --bind-to core /home/mwa32/atomate_g/VASP/vasp.6.4.2_WSM3/bin/vasp_std

