#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Check http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html
# Written by Shikai Jin on 2021-Nov-25, latest modified on 2021-Nov-25

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import sys
import platform
import sys
import pdbfixer

def fixPDB(pdb_file):
    """Uses the pdbfixer library to fix a pdb file, replacing non standard residues, removing
    hetero-atoms and adding missing hydrogens. The input is a pdb file location,
    the output is a fixer object, which is a pdb in the openawsem format."""
    fixer = pdbfixer.PDBFixer(filename=pdb_file, )
    a = fixer.findMissingResidues()

    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    return (fixer.missingResidues or (fixer.nonstandardResidues or fixer.missingAtoms))

pdb_file = sys.argv[1]
#print(pdb_file)
reject = fixPDB(pdb_file)

if reject:
    with open('reject.txt', 'a') as the_file:
        the_file.write(pdb_file[:-4] + '\n')