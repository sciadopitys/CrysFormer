#!/bin/bash
cd /data/georgep/Projects/interpret_retrain

# autobuilds using coordinates from fitting the fixed map, then uses 2FOFC mapos to refit better

source /programs/sbgrid.shrc

# loop over all test cases, or every Nth one.

for FILE in /data/georgep/Projects/interpret_retrain/*.pd_*; do
basename "$FILE"
filename="$(basename -- $FILE)"
echo "$filename"
prefix=$filename
cd "$prefix"
cp ./AutoBuild_run_2_/overall_best.pdb model.pdb
/programs/i386-mac/system/sbgrid_bin/phenix.autobuild model=model.pdb seq_file=seq.fasta data=amplitudes.mtz ncs_copies=1 find_ncs=False two_fofc_in_rebuild=True 
cd ../
done
