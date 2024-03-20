#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Written by Shikai Jin on 2021-Mar-12, latest modified on 2021-Aug-11
# A greatly expansion of the original clean pdb code, tested on Python 3.8
# Modified from a Python 2 version pdb-tools
# Example in Linux: python this_code.py input.pdb

import time
import string
import argparse

# Types of coordinate entries
COORD_RECORDS = ["ANISOU", "ATOM  ", "HETATM", "END   ", "ENDMDL", "MODEL", "TER   "]

# Problems with structure warranting user attention
ERROR_RECORDS = ["CAVEAT", "OBSLTE"]

AA_3_INDEX = ['ALA', 'CYS', 'GLU', 'PHE', 'GLY', 'HIS', 'HSE', 'HSD', 'ILE', 'LYS', 'LEU', 'MET', 'MES', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AA_1_INDEX = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

class PdbCleanError(Exception):
    """
    General exception to raise if there is a problem with this module.
    """

    pass


def pdbCheck(coord):
    """
    Make sure the pdb file still has something in it after processing.
    """

    pdb_check = len([l for l in coord if l[0:6] == "ATOM  "])

    if pdb_check > 0:
        return 0
    else:
        return 1


def convertModifiedAA(coord, header):
    """
    Convert modified amino acids to their normal counterparts.
    """
    # See if there are any non-standard amino acids in the pdb file.  If there
    # are not, return
    modres = [l for l in header if l[0:6] == "MODRES"]

    # If there is no MODRES remark directly return
    if len(modres) == 0:
        return coord, header, []

    # Create list of modified residues
    mod_dict = dict([(l[12:15], l[24:27]) for l in modres])

    # Convert to ATOM entries, skipping non-backbone atoms.  These will be built
    # with CHARMM.
    backbone_atoms = ["N  ", "CA ", "C  ", "O  "]
    new_coord = []
    for line in coord:
        if line[17:20] in list(mod_dict.keys()):
            new = mod_dict[line[17:20]]
            if line[13:16] in backbone_atoms:
                new_line = "ATOM  %s%s%s" % (line[6:17], new, line[20:])
                new_coord.append(new_line)
        else:
            new_coord.append(line)

    # Convert non-standard atoms in the SEQRES entries
    converted_list = []
    new_header = []
    for line in header:
        if line[0:6] == "SEQRES":
            old_seq = line[19:70].split()
            new_seq = []
            for aa in old_seq:
                if aa in list(mod_dict.keys()):
                    new_seq.append(mod_dict[aa])
                else:
                    new_seq.append(aa)

            new_seq = "".join(["%s " % aa for aa in new_seq])
            new_seq.strip()
            new_seq = "%-50s" % new_seq

            new_header.append("%s%-50s%s" % (line[:19], new_seq, line[71:]))
        else:
            new_header.append(line)

    # Create output remarks
    conv = ["REMARK       converted %s to %s\n" % (k, mod_dict[k])
            for k in list(mod_dict.keys())]

    return new_coord, new_header, conv


def stripACS(coord):
    """
    Removes alternate conformations.
    """

    def removeLetters(line):
        """
        Mini function that removes letters that denote ACS.
        """

        if line[16] in string.ascii_letters:
            line = "%s %s" % (line[:16], line[17:])  # Alternate location indicator
        if line[26] in string.ascii_letters:
            line = "%s %s" % (line[:26], line[27:])  # Code for insertions of residues
        return line

    # If a particular residue already has an atom, it will be in known_atom_dict
    # The second occurence of that atom in the same residue is assumed to be an
    # alternate conformation and is skipped.
    known_atom_dict = {}
    known_residue_name_dict = {}
    coord_out = []
    skipped = []
    for c in coord:
        residue = c[21:26]  # Chain plus residue index, no insertion code

        # If the residue is not known, update known_atom_dict and append line
        # to coordinate file
        if residue not in list(known_atom_dict.keys()):
            out = removeLetters(c)
            coord_out.append(out)
            known_atom_dict.update([(residue, [c[13:16]])])
            known_residue_name_dict.update([(residue, [c[17:20]])])

        # If the residue is known, determine if the atom has been seen before.
        # If it has, skip it.  Otherwise, 
        # append to coord_out and known_atom_dict
        else:
            atom = c[13:16]
            residue_name = c[17:20]
            if atom in known_atom_dict[residue]:
                skipped.append("REMARK%s" % c[6:])
            else:
                if residue_name in known_residue_name_dict[residue]:    
                    out = removeLetters(c)
                    coord_out.append(out)
                    known_atom_dict[residue].append(atom)
                else:
                    skipped.append("REMARK%s" % c[6:])

    return coord_out, skipped


def backboneCheck(coord):
    """
    Checks for duplicate residues (fatal) and missing backbone atoms.  If a
    backbone atom is missing, the entire containing residue is deleted.
    """

    residue_numbers = []
    for line in coord:
        if line[17:26] not in residue_numbers:
            residue_numbers.append(line[17:26]) # Include line[17-25], actually index 18-26 in one line

    to_remove = []
    for resid in residue_numbers: # Map all atoms to each "Resname Chain Resindex"
        resid_atoms = [l for l in coord if l[17:26] == resid]

        # All backbone atoms in the protein
        backbone_atoms = [[l for l in resid_atoms if l[13:16] == "N  "],
                          [l for l in resid_atoms if l[13:16] == "CA "],
                          [l for l in resid_atoms if l[13:16] == "C  "],
                          [l for l in resid_atoms if l[13:16] == "O  "]]

        # If this is a proline, add CD to required backbone atoms
        if resid[0:3] == "PRO":
            backbone_atoms.append([l for l in resid_atoms if l[13:16] == "CD "])

        # If more than one of a backbone atom is found for a residue, we have
        # some sort of duplication.  If a backbone atom is missing, delete the
        # residue.
        for b in backbone_atoms:
            if len(b) > 1:
                err = "\%s\" is duplicated!" % resid
                raise PdbCleanError(err)
            if len(b) == 0:
                to_remove.append(resid)

    coord = [l for l in coord if l[17:27] not in to_remove]
    removed = ["REMARK       removed %s\n" % r for r in to_remove]
    return coord, removed


def pdbAtomRenumber(pdb, renumber_het=True):
    """
    Renumber all atoms in pdb file, starting from 1.
    """

    entries_to_renumber = ["ATOM  ", "TER   ", "ANISOU"]
    if renumber_het == True:
        entries_to_renumber.append("HETATM")

    out = []
    counter = 1
    for line in pdb:
        # For and ATOM record, update residue number
        if line[0:6] in entries_to_renumber:
            out.append("%s%5s%s" % (line[0:6], counter, line[11:]))
            #print(counter)
            counter += 1
        else:
            # reset the counter for a new model
            if line[0:6] == "ENDMDL":
                counter = 1
            out.append(line)
    return out


def pdbResidueRenumber(coord, renumber_het=True):
    """
    Renumber all residues in pdb file, starting from 1. If multiple chains, index will go back to 1
    """

    entries_to_renumber = ["ATOM  ", "TER   ", "ANISOU"]
    if renumber_het == True:
        entries_to_renumber.append("HETATM")

    out = []
    resid_record = 0
    resid_old = 'NULL'
    for line in coord:
        # For and ATOM record, update residue number
        if line[0:6] in entries_to_renumber:
            resid = int(line[22:26])
            if resid != resid_old:
                resid_old = resid
                resid_record += 1
            out.append("%s%4d%s" % (line[0:22], resid_record, line[26:]))
        else:
            # reset the counter for a new model
            if line[0:6] == "ENDMDL":
                resid_record = 0
                resid_old = 'NULL'
            out.append(line)
    return out

def resetBfactor(coord, b_factor=20.0):
    """
    Reset all bfactors to the given value
    """
    entries_to_renumber = ["ATOM  ", "HETATM"]
    out = []
    for line in coord:
        # For and ATOM record, update residue number
        if line[0:6] in entries_to_renumber:
            out.append("%s%6.2f%s" % (line[0:60], b_factor, line[66:]))
        else:
            out.append(line)

    return out

def assignChain(coord, chain_id='A'):
    """
    Assign all atoms without chain id to A
    """
    entries_to_renumber = ["ATOM  ", "HETATM"]
    out = []
    for line in coord:
        # For and ATOM record, update residue number
        if line[0:6] in entries_to_renumber:
            out.append("%s%s%s" % (line[0:21], chain_id, line[22:]))
        else:
            out.append(line)
    return out


def pdbClean(pdb, pdb_id="temp", chains="all", renumber_atoms=False, renumber_residues=False,
             remove_alternate=False, standard_element='False', leave_atom="all", reset_bfactor=20.0, assign_chain='A', verbose=True):
    """
    Standardize a pdb file:
        - Remove waters, ligands, and other HETATMS
        - Convert modified residues (i.e. Se-Met) to the normal residue
        - Remove alternate confpdbAtomchoose(coord)ormations (taking first in pdb file)
        - Take only the specified chain
        - Renumber residues from 1
    """

    # Set up log
    #log = ["REMARK  PDB processed using clean_pdb.py \n"]
    #log_fmt = "REMARK   - %s\n"
    #log.append(log_fmt % ("Process time: %s" % time.asctime()))

    # Check pdb files for error warnings (CAVEAT and OBSLTE)
    error = [l for l in pdb if l[0:6] in ERROR_RECORDS]
    if len(error) != 0:  #
        # log = ["%-79s\n" % (l.strip()) for l in log]
        # try:
        #     remark_pos = [l[0:6] for l in header].index("REMARK")
        # except ValueError:
        #     remark_pos = 0
        err = "PDB might have problem!\n" + "".join(error)
        raise PdbCleanError(err)

    # Grab pdb header, excluding coordinates and deprecated records.
    header = [l for l in pdb if l[0:6] not in COORD_RECORDS]
    # [print(i, end="") for i in header]

    # Convert non-standard amino acids to standard ones
    coord = [l for l in pdb if l[0:6] in COORD_RECORDS]
    #coord, header, converted = convertModifiedAA(coord, header)

    # Strip all entries in COORD_RECORDS except ATOM
    coord = [l for l in coord if l[0:6] == "ATOM  "]
    if pdbCheck(coord):
        err = "There are no ATOM entries in this pdb file!"
        raise PdbCleanError(err)
    else:
        #log.append(log_fmt % "HETATM entries removed.")
        #if verbose:
        #    print(log[-1], end=' ')
        pass
    
    if pdbCheck(coord):
        err = "Chain filter (%r) removed all atoms in pdb file!" % chains
        raise PdbCleanError(err)

    if reset_bfactor != None:
        coord = resetBfactor(coord, b_factor=reset_bfactor)
    
    if assign_chain != None:
        coord = assignChain(coord, chain_id=assign_chain)

    # Check for missing backbone atoms; these residues are deleted
    coord, removed = backboneCheck(coord)
    if len(removed) != 0:
        #log.append(log_fmt % "Residues with missing backbone atoms removed.")
        #if verbose:
            #print(log[-1], end=' ')
        #log.extend(removed)
        pass
    if pdbCheck(coord):
        err = "Backbone checker removed all atoms!  Mangled pdb file."
        raise PdbCleanError(err)

    # Renumber atoms if requested
    if renumber_atoms:
        coord = pdbAtomRenumber(coord)
        # [print(i, end="") for i in coord]
        #log.append(log_fmt % "Atoms renumbered from one.")
        #if verbose:
            #print(log[-1], end=' ')

    # Renumber residues if requested
    if renumber_residues:
        coord = pdbResidueRenumber(coord)
        # [print(i, end="") for i in coord]
        #log.append(log_fmt % "Residues renumbered from one.")
        #if verbose:
            #print(log[-1], end=' ')

    # Leave the atoms that we want
    if leave_atom != "all":
        leave_atom = leave_atom.split(',')
        coord = [l for l in coord if l[77].strip() in leave_atom]  # Leave only the given type of atoms
        coord = pdbAtomRenumber(coord)
        #log.append(log_fmt % ("Leave the atoms in the given set %r." % leave_atom))
        #if verbose:
            #print(log[-1], end=' ')

    # Final check
    if pdbCheck(coord):
        err = "Unknown error occured and pdb has been mangled!"
        raise PdbCleanError(err)

    #log = ["%-79s\n" % (l.strip()) for l in log]

    try:
        remark_pos = [l[0:6] for l in header].index("REMARK")
    except ValueError:
        remark_pos = 0

    # Return processed pdb file, placing log after preliminary remarks.
    out_pdb = []
    # out_pdb.extend(header)
    # out_pdb.extend(log)
    out_pdb.extend(coord)

    # print(log)

    return out_pdb


def main():
    #########
    # Prepare the input options
    # Always
    parser = argparse.ArgumentParser(
        description="This script cleans the pdb file")
    parser.add_argument("input", help="The file name of input PDB file", type=str)
    parser.add_argument("--renumber_residues", help="Renumber residues", action="store_true", default=True)
    parser.add_argument("--renumber_atoms", help="Renumber atoms", action="store_true", default=True)
    parser.add_argument("--leave_atom", help="Leaving the given atoms", type=str, default='all')
    parser.add_argument("--reset_bfactor", help="Reset all B factor to the given value", type=float)
    parser.add_argument("--assign_chain", help="Assign the chain to all atoms", type=str, default='A')
    parser.add_argument("-v", "--verbose", help="Verbose mode", action="store_true", default=False)
    parser.add_argument("output", help="The file name of output pdb", type=str)
    args = parser.parse_args()
    input_file = args.input
    
    chains = 'all'
    
    renumber_atoms = args.renumber_atoms
    
    renumber_residues = args.renumber_residues
    
    remove_alternate = False
    
    leave_atom = args.leave_atom
    
    standard_element = False
    
    verbose = args.verbose
    
    reset_bfactor = args.reset_bfactor
    
    assign_chain = args.assign_chain
    
    output_file = args.output
    #########

    with open(input_file, 'r') as fread:
        content = []
        for line in fread.readlines():
            content.append(line)

    clean_data = pdbClean(content, pdb_id="temp", chains=chains, renumber_atoms=renumber_atoms, renumber_residues=renumber_residues,
                          remove_alternate=remove_alternate, standard_element=standard_element, leave_atom=leave_atom, reset_bfactor=reset_bfactor, assign_chain=assign_chain, verbose=verbose)

    with open(output_file, 'w') as fwrite:
        for line in clean_data:
            fwrite.writelines(line)


if __name__ == "__main__":
    main()
