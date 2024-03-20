#!/bin/bash
# this script prepares the needed files for autobuild

cd /data/georgep/Projects/interpret_retrain

# loop over all test cases in the directory of predictions

source /programs/sbgrid.shrc
for FILE in /data/qp3/interpret_retrain/*.pd_*; do
basename "$FILE"
filename="$(basename -- $FILE)"
echo "$filename"
	prefix=$filename
 
# make directory if it does not exist and cd to it

mkdir /data/georgep/Projects/interpret_retrain/"$prefix"
        cd "$prefix"

# make Fasta sequence file from 3-letter file

sed 's/ALA/A/g;s/CYS/C/g;s/ASP/D/g;s/GLU/E/g;s/PHE/F/g;s/GLY/G/g;s/HIS/H/g;s/HID/H/g;s/HIE/H/g;s/ILE/I/g;s/LYS/K/g;s/LEU/L/g;s/MET/M/g;s/ASN/N/g;s/PRO/P/g;s/GLN/Q/g;s/ARG/R/g;s/SER/S/g;s/THR/T/g;s/VAL/V/g;s/TRP/W/g;s/TYR/Y/g;s/MSE/X/g' < /data/qp3/interpret_retrain/$prefix/${prefix}_seq.txt  | tr -d " " | awk -v prefix=$prefix 'BEGIN{print ">P1;",prefix} length($0)>1{print}' > seq.fasta

# cp mtz file for predicted map

cp /data/qp3/interpret_retrain/$prefix/${prefix}_prednew.mtz ./prednew.mtz

# make mtz file for F and sigma 

/programs/i386-mac/system/sbgrid_bin/sftools << eof
read /data/qp3/interpret_retrain/$prefix/${filename}_act.mtz
calc col SIGFP = 1
set type col SIGFP
Q
set label col FMODEL
FP
calc col FreeR_flag = Rfree(0.05)
set type col FreeR_flag
I       
CALC col FP = col FP 10 *
write amplitudes.mtz MTZ col FP SIGFP FreeR_flag
quit
eof

	cd ../
done
