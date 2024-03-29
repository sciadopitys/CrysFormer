#!/bin/bash 
# to launch use -
# source /programs/x/ccp4/8.0/ccp4-8.0/bin/ccp4.setup-sh 
# export PATH=/programs/x/shelx-2013/2013/bin:$PATH 
# time (cat  /data/qp3/interpret_15_angle_training_newrange/list.txt | nice -n 5  parallel -j30  ./setup-shelxe-retrain-2.sh {}) &
prefix=$1
mkdir -p $prefix
cd $prefix
#/programs/x86_64-linux/phenix/1.20.1-4487/phenix-1.20.1-4487/build/bin/phenix.print_sequence /data/qp3/interpret_retrain/$prefix/${prefix}.pdb | awk 'NR==1{print ">P1;",substr($0,2);print ""}NR>1&&length($0)>1{print}' > $prefix.seq
/programs/x86_64-linux/ccp4/8.0/ccp4-8.0/bin/sftools << eof > sftools-$prefix.log
read  /data/qp3/interpret_15_angle_training_newrange/${prefix}_act.mtz
calc col SIGFP = 1
set type col SIGFP
Q
set label col FMODEL
FP
delete col PHIFMODEL
read   /data/qp3/interpret_15_angle_training_newrange/${prefix}_prednew.mtz
calc col FreeR_flag = Rfree(0.05)
set type col FreeR_flag
I	
calc col FP = col FP 10 *
calc col FWT = col FWT 10 *
write ${prefix}_startARP.mtz
calc col FOM = 0.5
set type col FOM 
W
write ${prefix}_start.phs PHS col FWT FOM PHWT 
correl col FP  FWT
quit
eof

/programs/x86_64-linux/ccp4/8.0/ccp4-8.0/bin/mtz2hkl  ${prefix}_startARP.mtz  > /dev/null
ln -s    ${prefix}_startARP.hkl $prefix.hkl
#pdb2ins  /data/qp3/interpret_retrain/$prefix/$prefix.pdb -w 1 -h 4 -i -o tmp.ins
/programs/x86_64-linux/shelx-2013/2013/bin/pdb2ins /data/qp3/interpret_15_angle_training_newrange/$prefix.pdb -w 1 -h 4 -i -o tmp.ins > /dev/null
head -22 tmp.ins >  $prefix.ins
#head -22 /data/phillips/mitchm/for-Tom/arp-warp-test/7MPZ_1.pd_61/shelxe/tmp.ins  >   $prefix.ins
solvent=0.8
#solvent=`/programs/x86_64-linux/ccp4/8.0/ccp4-8.0/bin/gemmi contents /data/qp3/interpret_retrain/$prefix/$prefix.pdb | awk '/protein density 1.34/{print $NF/100}'`
ln -s ${prefix}_start.phs ${prefix}.phi
 
#/data/phillips/mitchm/software-downloads/shelx/linux64/shelxe ${prefix}.phi -a8 -s$solvent > run-shelxe.log
/programs/x/shelx-2013/2013/bin/shelxe ${prefix}.phi -a4 -s$solvent > run-shelxe.log

