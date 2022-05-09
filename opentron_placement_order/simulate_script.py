

import sys
from opentrons.simulate import simulate, format_runlog
import re
import os


def get_aspirate_locs(script_file):
    # extract the locations of R1, R1, I1 from the file
    protocol_file = open(script_file)
    R1 = 'False'
    R2 = 'False'
    I1 = 'False'

    for line in protocol_file:
        if 'R1= ' in line or 'R1 =' in line:
            loc = re.findall("'[A-Z][0-9]+'", line)[0]
            R1 = loc[1:-1]
        elif 'R2= ' in line or 'R2 =' in line:
            loc = re.findall("'[A-Z][0-9]+'", line)[0]
            R2 = loc[1:-1]
        elif 'I1= ' in line or 'I1 =' in line:
            loc = re.findall("'[A-Z][0-9]+'", line)[0]
            I1 = loc[1:-1]

        if any(i in line for i in ['R1=', 'R1 =', 'R2=', 'R2 =', 'I1=', 'I1 =']):
            print(line)

    return R1, R2, I1

def append_dispense_data(script_file, R1,R2,I1):
    protocol_file = open(script_file)
    path, script_name = os.path.split(script_file)
    path, folder = os.path.split(path)


    out_file = open('dispense_data.csv', 'a')
    # simulate() the protocol, keeping the runlog
    runlog, _bundle = simulate(protocol_file)

    t_tip = 0
    t_aspirate = 0

    for line in format_runlog(runlog).splitlines():
        print()
        print(line)
        if 'Picking up' in line:
            print('new tip')
            t_tip = 0

        if 'Aspirating' in line:
            print('R1: {}, R2:{}, I1:{}'.format(R1, R2, I1))
            t_aspirate = 0
            if R1 in line:
                filled_with = 'R1'
            elif R2 in line:
                filled_with = 'R2'
            elif I1 in line:
                filled_with = 'I1'
            print('Aspirating: ' + filled_with)

        if 'Dispensing' in line:
            t_tip += 1
            t_aspirate += 1
            plate = re.findall('Flat on [0-9]',line)[0][-1]
            loc = re.findall('[A-Z][0-9]+', line)[0]

            print('dispensing {} at {} on {}. t_aspirate: {} t_tip: {}'.format(filled_with, loc, plate, str(t_aspirate), str(t_tip)))
            out_file.write('\n{},{},{},{},{},{},{}'.format(script_name, folder, filled_with, plate, loc, str(t_aspirate), str(t_tip)))

script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/0x3D_ZG-ZBD/220330_2receiver.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/0x7F_ZG/211214_ZG_0x7F_rep2.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/0x37_ZG/220302_ZG_0x37-D2.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/AND_ZG-Diagonal-2/211118_ZG_2_Diagonal.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/majority_ZG/220303_ZG_majority-D2.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/OR_ZG-Horizontal-1/220120_ZG_1_Horizontal_rep2.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/sum1_ZBD/211214_ZBD_sum1_rep3.py'
script_file = '/Users/neythen/Desktop/Projects/synbiobrain/opentron_placement_order/opentron_protocols/XOR_ZBD-Diagonal-1/220120_ZBD_1_Diagonal_rep2.py'



R1, R2, I1 = get_aspirate_locs(script_file)
print('R1: {}, R2:{}, I1:{}'.format(R1,R2,I1))

append_dispense_data(script_file, R1, R2, I1)





