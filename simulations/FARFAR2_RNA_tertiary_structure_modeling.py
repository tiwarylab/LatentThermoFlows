import os
import argparse
import mdtraj as md
import numpy as np

def parse_args():
    parse = argparse.ArgumentParser(description="assemble and optimize 3D RNA structures via 3dRNA")    
    parse.add_argument("-num", "--number", default="10", help='the number of assemble structures')
    parse.add_argument("-ss", "--secondary_structure", required=True, help='input RNA secondary structure -dot-braket format-')
    parse.add_argument("-r", "--rounds", default=3, help="the number of assembled and optimized structures")
    parse.add_argument("-c", "--cycles", default=2000, help="the number of steps to run Monte Carlo")
    parse.add_argument("-t", "--temperature", default=2000, help="the temperature to run Monte Carlo")
    args = parse.parse_args()
    return args

args = parse_args()
with open("../secondary_structure_ensemble/structure_{}.seq".format(args.secondary_structure), 'r') as f:
    _secondary_structure = f.readlines()[1][:-1]

for num in range(int(args.number)):
    ##run the farfar2 via rosetta to assemble the RNA structures and run the Monte Carlo
    os.system('rna_denovo.static.linuxgccrelease -sequence ggcgcaagcc -secstruct \"'+_secondary_structure +'\" -minimize_rna true -nstruct 1 -rna:denovo:rounds '+args.rounds+' -cycles '+args.cycles+'  -rna:denovo:temperature '+args.temperature+' -rna:denovo:minimize:minimize_bps true -rna:denovo:bps_moves true -out:overwrite true -out:file:silent output_temp_'+args.temperature+'.temp')
    temp_filename = "output_temp_"+args.temperature+".temp"; out_filename = "final_output_temp_"+args.temperature+".out"
    if not os.path.isfile(temp_filename):
        os.rename(temp_filename, out_file_name)
    else:
        with open(temp_filename, 'r') as temp_file, open(out_filename, 'a') as out_file:
            for line in temp_file:
                out_file.write(line)
        os.remove(temp_filename)

## transfer the output file into the pdb and save the energy files
os.system('extract_pdbs.static.linuxgccrelease -in:file:silent final_output_temp_'+args.temperature+'.out -in:file:silent_struct_type nucleic')
energies = []
with open(out_filename, 'r') as file:
    for line in file:
        if line.startswith("SCORE:"):
            parts = line.split()
            if len(parts) > 1 and parts[1].lower() != "score":
                try:
                    score = float(parts[1])
                    energies.append(score)
                except ValueError:
                    print(f"Warning: Could not convert '{parts[1]}' to float.")
                    energies.append(np.nan)

