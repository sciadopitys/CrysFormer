#!/bin/bash -l

#SBATCH --job-name=step7
#SBATCH --array=1-20%20 # default is 10
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time=144:00:00
#SBATCH --mail-user=Shikai.Jin@rice.edu
#SBATCH --mail-type=FAIL

date
echo My job ran on: $SLURM_NODELIST
echo My jobid was $SLURM_JOBID
echo My submission directory was $SLURM_SUBMIT_DIR

start=`echo "($SLURM_ARRAY_TASK_ID-1)*1350+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*1350" | bc`

for j in {0..19}
do
    mkdir 7_electron_density_pt_${j}
    python3 convert_ccp4_to_pt_ed.py split_${j}.txt ${j} &
done
