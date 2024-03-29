This script is to take the output prediction from CrysFormer and use it as an input
map for density modification and poly-Alanine autotracing using shelxe.  The script
requires shex package and the ccp4 programs. First sftools (from CCP4) is used to 
convert the mtz file into xrayview PHS format as expected by shelxe. The shelxe 
.ins file is genereated from the header lins of pdb2ins from the shelx package
This helps to set the correct unit cell etc. input files needed are the 
ground truth amplitudes (used as Fobs), the ground truth .pdb file (used
to get the unit cell etc in the shelxe input header), and the .mtz file
created from the CrystFormer pytorch map file. To run in parallel, 
gnu-parallel can be used. The script expects as input the file prefix.
I then makes a subdirecty from that input, creatres the input files needed
to run shelxe starting from the map coefficients. 
time (cat  /data/qp3/interpret_15_angle_training_newrange/list.txt | nice -n 5  parallel -j30  ./setup-shelxe-retrain.sh {}) &
