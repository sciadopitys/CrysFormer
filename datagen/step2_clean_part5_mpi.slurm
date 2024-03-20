#!/bin/bash -l

#SBATCH --job-name=step2
#SBATCH --array=1-50%50 # default is 10
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
start=`echo "($SLURM_ARRAY_TASK_ID-1)*7947+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*7947" | bc`

# cat full_pdbID_list.txt | sed -n "$start ,$end p"

# if [ $SLURM_ARRAY_TASK_ID == 1 ];
#   then start=1
# fi

mkdir 2_pdb_reweight_allatom_clean_reorient
mkdir 2_pdb_reweight_allatom_clean_reorient_center
cd 2_pdb_reweight_allatom_clean
for i in `cat ../step1_full_id_list.txt | sed -n "$start ,$end p"`; do sed -i '/CRYST1/d' ${i}.pdb; sed -i '1i\CRYST1   41.000   30.000   24.000  90.00  90.00  90.00 P 1           1       ' ${i}.pdb;done
for i in `cat ../step1_full_id_list.txt | sed -n "$start ,$end p"`; do /usr/bin/python3 ../xyz_reorient.py ${i}.pdb ../2_pdb_reweight_allatom_clean_reorient/${i}.pdb;done
cd ../2_pdb_reweight_allatom_clean_reorient
for i in `cat ../step1_full_id_list.txt | sed -n "$start ,$end p"`; do sed -i '/CRYST1/d' ${i}.pdb; sed -i '1i\CRYST1   41.000   30.000   24.000  90.00  90.00  90.00 P 1           1       ' ${i}.pdb;done
for i in `cat ../step1_full_id_list.txt | sed -n "$start ,$end p"`; do /usr/bin/python3 ../pdb_translation_com.py ${i}.pdb ../2_pdb_reweight_allatom_clean_reorient_center/${i}.pdb;done
cd ..
