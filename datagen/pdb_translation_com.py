import sys
import numpy as np
input_pdb = sys.argv[1]
output_pdb = sys.argv[2]

data = []
atom_weight = []
xyz_coord = []
com_position = [20.5, 15.0, 12.0]
with open(input_pdb, 'r') as fopen:
    lines = fopen.readlines()
    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            data.append(line)

for each_atom in data:
    xyz_coord.append([float(each_atom[30:38]), float(each_atom[38:46]), float(each_atom[46:54])])
    if each_atom[77] == 'N':
        atom_weight.append(14)
    elif each_atom[77] == 'O':
        atom_weight.append(16)
    elif each_atom[77] == 'C':
        atom_weight.append(12)
    elif each_atom[77] == 'S':
        atom_weight.append(32)
    elif each_atom[77] == 'H':
        atom_weight.append(1)
    else:
        atom_weight.append(0)


xyz_coord = np.array(xyz_coord)
xyz_mean = np.average(xyz_coord, axis=0, weights=atom_weight)
print(xyz_mean)
xyz_shift = xyz_mean - com_position

new_data = []
for line in lines:
    if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
        x = float(line[30:38]) - xyz_shift[0]
        y = float(line[38:46]) - xyz_shift[1]
        z = float(line[46:54]) - xyz_shift[2]
        new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
    else:
        new_data.append(line)

with open(output_pdb, 'w') as fwrite:
    for new_each_atom in new_data:
        fwrite.writelines(new_each_atom) 
