#!/bin/bash
# full default autobuild procedure

cd /data/georgep/Projects/interpret_retrain
source /programs/sbgrid.shrc

# loop over all test cases

for FILE in /data/georgep/Projects/interpret_retrain/*.pd_*; do
basename "$FILE"
filename="$(basename -- $FILE)"
echo "$filename"
prefix=$filename
cd "$prefix"
/programs/i386-mac/system/sbgrid_bin/phenix.autobuild seq_file=seq.fasta data= amplitudes.mtz map_file=prednew.mtz 
cd ../
done
