#!/bin/bash -l

#SBATCH --job-name=step4
#SBATCH --array=1-20%20
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL
#SBATCH --time=144:00:00
#SBATCH --mail-user=Shikai.Jin@rice.edu
#SBATCH --mail-type=FAIL

date
echo My job ran on: $SLURM_NODELIST
echo My jobid was $SLURM_JOBID
echo My submission directory was $SLURM_SUBMIT_DIR

start=`echo "($SLURM_ARRAY_TASK_ID-1)*9804+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*9804" | bc`

source /path/to/ccp4-<version>/bin/ccp4.setup-sh
for j in {0..19}
do
    export split=${j}
    mkdir 4_patterson_ccp4_res_${j}
    cat split_${j}.txt | parallel -j 25 'fft hklin 3_electron_density_mtz_res_${split}/{}.gemmi.mtz mapout 4_patterson_ccp4_res_${split}/{}_patterson.ccp4 < command_step4.txt > 4_patterson_ccp4_res_${split}/{}_patterson.log'
done
