#!/bin/bash
cd /data/georgep/Projects/interpret_retrain

# autobuld using a fixed map to fit against, NCS off to sabve a bit of time

source /programs/sbgrid.shrc
# loop over all test cases
for FILE in /data/georgep/Projects/interpret_retrain/*.pd_*; do
basename "$FILE"
filename="$(basename -- $FILE)"
echo "$filename"
prefix=$filename
cd "$prefix"
/programs/i386-mac/system/sbgrid_bin/phenix.autobuild seq_file=seq.fasta data= amplitudes.mtz map_file=prednew.mtz ncs_copies = 1 find_ncs = False use_constant_input_map = True
cd ../
done
