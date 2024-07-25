#!/bin/bash -l

#SBATCH --job-name=step8
#SBATCH --array=1-20%20 # default is 10
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time=144:00:00
#SBATCH --mail-user=Shikai.Jin@rice.edu
#SBATCH --mail-type=END

date
echo My job ran on: $SLURM_NODELIST
echo My jobid was $SLURM_JOBID
echo My submission directory was $SLURM_SUBMIT_DIR

start=`echo "($SLURM_ARRAY_TASK_ID-1)*1350+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*1350" | bc`

mkdir ../patterson_pt_scaled
for j in {0..19}
do
    python3 scaling_tensor_pat.py split_${j}.txt 1045.0 -47.0 ${j} &
done
