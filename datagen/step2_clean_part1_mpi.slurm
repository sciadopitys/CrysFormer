#!/bin/bash -l

#SBATCH --job-name=step2
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

# source /home/sj52/program/ccp4-7.1/bin/ccp4.setup-sh
start=`echo "($SLURM_ARRAY_TASK_ID-1)*9001+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*9001" | bc`

# cat full_pdbID_list.txt | sed -n "$start ,$end p"

# if [ $SLURM_ARRAY_TASK_ID == 1 ];
#   then start=1
# fi

source /programs/sbgrid.shrc
cd 2_pdb_reweight
for i in `cat ../step1_full_id_list.txt | sed -n "$start ,$end p"`; do /programs/x/openmm/7.5.1/miniconda/bin/python ../allatom_prepare.py ${i}.pdb ;done 
