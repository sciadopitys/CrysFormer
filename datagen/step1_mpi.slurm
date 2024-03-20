#!/bin/bash -l

#SBATCH --job-name=step1
#SBATCH --array=1-1%1 # default is 10
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks=1
#SBATCH --mem=128G
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
start=`echo "($SLURM_ARRAY_TASK_ID-1)*5000+1" | bc`
end=`echo "($SLURM_ARRAY_TASK_ID)*5000" | bc`

# cat full_pdbID_list.txt | sed -n "$start ,$end p"

# if [ $SLURM_ARRAY_TASK_ID == 1 ];
#   then start=1
# fi

mkdir 2_pdb_reweight
cd /data/phillips/mitchm/for-Tom/new-raw/entities

input="/data/qp3/dataset_15_ext2/ids2.txt"
while IFS= read -r line
do
    python3 /data/qp3/dataset_15_ext2/convert_pdb_to_fifteen_aa_12sep.py ${line}.pdb.gz /data/qp3/dataset_15_ext2/2_pdb_reweight
  
done < "$input"