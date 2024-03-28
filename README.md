Dataset Generation:
Extract the files in pdb.tar.bz into 2_pdb_reweight_allatom_clean_reorient_center.  Then, run all step*_mpi.slurm scripts in the datagen directory in order.

Training:
For the initial training run, use "python3 train_initial.py".  To generate model predictions, use "python3 get_predictions.py".  Afterwards, for the recycling training run(s), use "python3 train_recycle.py". 
