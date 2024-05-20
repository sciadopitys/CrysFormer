Dataset Generation:

Extract the files in pdb.tar.bz into 2_pdb_reweight_allatom_clean_reorient_center.  Then, run all step*_mpi.slurm scripts in the datagen directory in order.  Steps 3, 4, 5 require that the ccp4 program suite is installed, as they make use of ccp4 command-line tools.

Training:

For the initial training run, use "python3 train_initial.py".  To generate model predictions, use "python3 get_predictions.py".  Afterwards, for the recycling training run(s), use "python3 train_recycle.py". 
