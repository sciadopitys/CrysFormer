#!/bin/bash
cd /data/georgep/Projects/interpret_retrain

#make default denmod file

grep "FREE R VALUE    " *.pd_*/*1*/overall_best.pdb | tr -s "/" " " | awk '{print $1, $NF}' >Rtest1
for i in $( cat Rtest1 | cut -f1 -d\  ) ;do
 echo `grep $i Rtest1 ` `grep $i pearson_subset.txt | awk '{print $1}'`  >>gather_denmod.txt
done

# make orig map only file

grep "FREE R VALUE    " *.pd_*/*2*/overall_best.pdb | tr -s "/" " " | awk '{print $1, $NF}' >Rtest2
for i in $( cat Rtest2 | cut -f1 -d\  ) ;do
 echo `grep $i Rtest2 ` `grep $i pearson_subset.txt | awk '{print $1}'`  >>gather_fixedmap.txt
done

# make 2fo-fc evolved  map ile

#grep "FREE R VALUE    " */*3*/overall_best.pdb | tr -s "/" " " | awk '{print $1, $NF}' >Rtest3
grep "FREE R VALUE    " *.pd_*/2fofc_final.pdb | tr -s "/" " " | awk '{print $1, $NF}' >Rtest3
for i in $( cat Rtest3 | cut -f1 -d\  ) ;do
 echo `grep $i Rtest3 ` `grep $i pearson_subset.txt | awk '{print $1}'`  >>gather_2fofc.txt
done
rm Rtest1
rm Rtest2
rm Rtest3
