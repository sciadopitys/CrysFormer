#!/bin/bash -l

#SBATCH --job-name=step3
#SBATCH --array=1-20%20 # default is 10
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time=24:00:00
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
    mkdir 3_electron_density_mtz_res_${j}
    cat split_${j}.txt | parallel -j 25 'gemmi sfcalc --dmin 1.5 --to-mtz=3_electron_density_mtz_res_${split}/{}.gemmi.mtz ../2_pdb_reweight_allatom_clean_reorient_center/{}.pdb'
done
