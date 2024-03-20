import sys
import numpy as np
import gemmi

data = []
xyz_coord = []
new_data = []

input_file = sys.argv[1]
output_file = sys.argv[2]


with open(input_file, 'r') as fopen:
    lines = fopen.readlines()
    cryst1 = lines[0]
    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            data.append(line)

for each_atom in data:
    xyz_coord.append([float(each_atom[30:38]), float(each_atom[38:46]), float(each_atom[46:54])])

xyz_range = np.amax(xyz_coord, axis=0) - np.amin(xyz_coord, axis=0)

ax_x = xyz_range[0]
ax_y = xyz_range[1]
ax_z = xyz_range[2]

x = str(round(float(cryst1[6:15]), 3))
ex = ' ' * (9 - len(x))
y = str(round(float(cryst1[15:24]), 3))
ey = ' ' * (9 - len(y))
z = str(round(float(cryst1[24:33]), 3))
ez = ' ' * (9 - len(z)) 
alpha = str(90.0)
ealpha = ' ' * (7 - len(alpha)) 
beta = str(90.0)
ebeta = ' ' * (7 - len(beta)) 
gamma = str(90.0)
egamma = ' ' * (7 - len(gamma)) 

cryst = ""
if ax_x >= ax_y:
    if ax_y >= ax_z:
        cryst = "CRYST1   " + x + ex + y + ey + z + ez + alpha + ealpha + beta + ebeta + gamma + egamma  + " P 1           1       "
        for line in lines:
            if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
            else:
                continue
    elif ax_x >= ax_z:
        cryst = "CRYST1   " + x + ex + z + ez + y + ey + alpha + ealpha + beta + ebeta + gamma + egamma  + " P 1           1       "  
        for line in lines:
            if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
                #LSM: (x,y,z) -> (x, z, len(S) - y)
                x = float(line[30:38])
                y = float(line[46:54])
                z = ax_y - float(line[38:46]) 
                new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
            else:
                continue                    
    else:
        cryst = "CRYST1   " + z + ez + x + ex + y + ey + alpha + ealpha + beta + ebeta + gamma + egamma  + " P 1           1       "   
        for line in lines:
            if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
                #MSL: (x,y,z) -> (z, len(M) - x, len(S) - y)
                x = float(line[46:54])
                y = ax_x - float(line[30:38])
                z = ax_y - float(line[38:46]) 
                new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
            else:
                continue                     
elif ax_y >= ax_z:
    if ax_x >= ax_z:
        cryst = "CRYST1   " + y + ey + x + ex + z + ez + alpha + ealpha + beta + ebeta + gamma + egamma  + " P 1           1       "
        for line in lines:
            if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
                #MLS: (x,y,z) -> (y, len(M) - x, z)
                x = float(line[38:46])
                y = ax_x - float(line[30:38])
                z = float(line[46:54])
                new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
            else:
                continue  
    else:
        cryst = "CRYST1   " + y + ey + z + ez + x + ex + alpha + ealpha + beta + ebeta + gamma + egamma  + " P 1           1       "
        for line in lines:
            if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
                #SLM: (x,y,z) -> (y, len(M) - z, len(S) - x) 
                x = float(line[38:46])
                y = ax_z - float(line[46:54])
                z = ax_x - float(line[30:38])
                new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
            else:
                continue 
else:
    cryst = "CRYST1   " + z + ez + y + ey + x + ex + alpha + ealpha + beta + ebeta + gamma + egamma  + " P 1           1       "
    for line in lines:
        if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
            #SML: (x,y,z) -> (z,y, len(S) - x)
            x = float(line[46:54])
            y = float(line[38:46])
            z = ax_x - float(line[30:38])
            new_data.append("%s%8.3f%8.3f%8.3f%s" %(line[:30], x, y, z, line[54:]))
        else:
            continue 

with open(output_file, 'w') as fwrite:
    fwrite.write(cryst + "\n")
    for new_each_atom in new_data:
        fwrite.writelines(new_each_atom) 
