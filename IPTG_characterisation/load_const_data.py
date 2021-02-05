import numpy as np
import matplotlib.pyplot as plt


def load_const_data(filepath, n_points):
    # dictionary structure: time_courses = dict[concentration]

    BP_data = {}
    TH_data = {}

    with open(filepath) as file:
        file.readline()
        for i, line in enumerate(file):
            line = line.split(',')

            time_point = int(line[1]) * 20.0  # each timepoint is 20mins
            flouresence = float(line[4])
            IPTG_conc = float(line[-1])
            name = line[-3]


            if name == '"ZBD"':
                try:
                    BP_data[IPTG_conc].append(flouresence)
                except:
                    BP_data[IPTG_conc] = []
                    BP_data[IPTG_conc].append(flouresence)
            elif name == '"ZG"':
                try:
                    TH_data[IPTG_conc].append(flouresence)
                except:
                    TH_data[IPTG_conc] = []
                    TH_data[IPTG_conc].append(flouresence)

    for IPTG_conc in TH_data.keys():
        timecourses = TH_data[IPTG_conc]

        # split the different repeats up

        repeats = []
        i = 0

        while (i + 1) * n_points <= len(timecourses):
            repeat = timecourses[i * n_points:(i + 1) * n_points]
            repeats.append(repeat)

            i += 1

        TH_data[IPTG_conc] = repeats

    for IPTG_conc in BP_data.keys():
        timecourses = BP_data[IPTG_conc]

        # split the different repeats up

        repeats = []
        i = 0
        while (i + 1) * n_points <= len(timecourses):
            repeat = timecourses[i * n_points:(i + 1) * n_points]
            repeats.append(repeat)

            i += 1

        BP_data[IPTG_conc] = repeats

    return TH_data, BP_data

if __name__ == '__main__':

    filepath= '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201202_IPTGagar_img_data_summary.csv'
    n_points = 64

    TH_data, BP_data = load_const_data(filepath, n_points)

    print(TH_data.keys())

    for IPTG_conc in [0., 1., 2.5, 5., 10., 50.]:
        plt.figure()
        plt.plot(np.array(TH_data[IPTG_conc]).T)
    plt.show()





