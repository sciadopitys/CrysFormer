#!/bin/bash -l

#SBATCH --job-name=step5
#SBATCH --array=1-20%20
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mail-user=Shikai.Jin@rice.edu
#SBATCH --mail-type=END

date
echo My job ran on: $SLURM_NODELIST
echo My jobid was $SLURM_JOBID
echo My submission directory was $SLURM_SUBMIT_DIR


start=`echo "($SLURM_ARRAY_TASK_ID-1)*9804+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*9804" | bc`

source /path/to/ccp4-<version>/bin/ccp4.setup-sh

for k in {0..19}
do
    export split=${k}
    mkdir 5_electron_density_ccp4_res_${k}
    cat split_${k}.txt | parallel -j 25 'echo -e "LABIN F1=FC PHI=PHIC \n GRID SAMPLE 3.0 \n" | fft hklin 3_electron_density_mtz_res_${split}/{}.gemmi.mtz mapout 5_electron_density_ccp4_res_${split}/{}_fft.ccp4'
done
