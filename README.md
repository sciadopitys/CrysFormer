# CrysFormer: Protein Structure Determination via Patterson Maps, Deep Learning and Partial Structure Attention

## Description
Determining the atomic-level structure of a protein has been a decades-long challenge. 
However, recent advances in transformers and related neural network architectures have enabled researchers to significantly improve solutions to this problem. 
These methods use large datasets of sequence information and corresponding known protein template structures, if available.
Yet, such methods only focus on sequence information. 
Other available prior knowledge could also be utilized, such as constructs derived from X-ray crystallography experiments and the known structures of the most common conformations of amino acid residues, which we refer to as partial structures. 
To the best of our knowledge, we propose the first transformer-based model that directly utilizes experimental protein crystallographic data and partial structure information to calculate electron density maps of proteins. 
In particular, we use Patterson maps which can be directly obtained from X-ray crystallography experimental data, thus bypassing the well-known crystallographic phase problem. 
We demonstrate that our method, \texttt{CrysFormer}, achieves precise predictions on two synthetic datasets of peptide fragments in crystalline forms, one with two residues per unit cell and the other with fifteen. 
These predictions can then be used to generate accurate atomic models using established crystallographic refinement programs.

## Dataset Generation:

Extract the files in pdb.tar.bz into 2_pdb_reweight_allatom_clean_reorient_center.  Then, run all step*_mpi.slurm scripts in the datagen directory in order.  Steps 3, 4, 5 require that the ccp4 program suite (https://www.ccp4.ac.uk/) is installed, as they make use of ccp4 command-line tools.  Replace the "source /path/to/ccp4-<version>/bin/ccp4.setup-sh" line in those scripts with a line corresponding to the installed version of ccp4.

## Environment/Dependencies:
Requires Torch version >= 1.12.0 and the einops version >= 0.6.0.

## Training:

For the initial training run, call "python3 train_initial.py".  To generate model predictions from the initial training run, call "python3 get_predictions.py".  Afterwards, for the recycling training run, call "python3 train_recycle.py".
