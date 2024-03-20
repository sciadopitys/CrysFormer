import sys
import gzip
data = {}
pdb_file = sys.argv[1]
output_dir = sys.argv[2]

with gzip.open(pdb_file, 'rt') as fopen:
    for line in fopen:
        if line[0:6].strip() == 'ATOM':
            atom_name = line[12:16].strip()
            try:
                residue_index = int(line[22:27])
                chain_index = line[21]
                try:
                    data[str(residue_index)+chain_index].append(line)
                except:
                    data[str(residue_index)+chain_index]=[line]
            except:
                pass
        elif line[0:6].strip() == 'ENDMDL':
            break
            
input_prefix_list = pdb_file[:-4].split('/')
input_prefix = input_prefix_list[-1]
flag = 0 # Detect whether we got a data point
keys = list(data.keys())
for i in range(0, len(keys)):
    try:
        if flag > 0:
            flag -= 1
            continue
        for check_label in range(1, 16):
            assert int(keys[i+check_label][:-1]) - int(keys[i+check_label-1][:-1]) == 1
        with open('%s/%s_%d.pdb' %(output_dir, input_prefix, i+1), 'w') as fwrite:
            for increment in range(0, 15):
                fwrite.writelines(data[keys[i+increment]])
        flag = 11 # set 11 but actually separate 12 times, since the first time is already counted

    except:
        pass
