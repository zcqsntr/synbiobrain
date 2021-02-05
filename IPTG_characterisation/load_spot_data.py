import numpy as np
import matplotlib.pyplot as plt


def load_spot_data(filepath, n_points):
    # dictionary structure: time_courses = dict[spot_concentration][distance]
    observed_wells = []  # to keep track of when we start a new repeat

    data = {}
    with open(filepath) as file:
        file.readline()
        for i, line in enumerate(file):
            line = line.split(',')

            time_point = int(line[1]) * 20.0  # each timepoint is 20mins
            flouresence = float(line[4])
            IPTG_conc = float(line[-2])
            distance = float(line[-1])

            try:
                data[IPTG_conc][distance].append(flouresence)
            except:
                try:

                    data[IPTG_conc][distance] = []
                    data[IPTG_conc][distance].append(flouresence)
                except:

                    data[IPTG_conc] = {}
                    data[IPTG_conc][distance] = []
                    data[IPTG_conc][distance].append(flouresence)

    # print(data)
    for IPTG_conc in data.keys():


        for d, distance in enumerate(data[IPTG_conc].keys()):
            timecourses = data[IPTG_conc][distance]

            # split the different repeats up

            repeats = []
            i = 0

            while (i + 1) * n_points <= len(timecourses):


                repeat = timecourses[i * n_points:(i + 1) * n_points]
                repeats.append(repeat)

                i += 1

            data[IPTG_conc][distance] = repeats
    return data

if __name__ == '__main__':


    filepath_BP = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201124_IPTGsendersZBD_img_data_summary.csv'
    n_points = 67

    colours = ['b', 'red', 'g']

    data_BP = load_spot_data(filepath_BP, n_points)
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Bandpass: ' + str(IPTG_conc))
        for i,distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(data_BP[IPTG_conc][distance]).T, colours[i])



    filepath_TH = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201201_IPTGsendersZG_img_data_summary.csv'
    n_points = 62
    data_TH = load_spot_data(filepath_TH, n_points)
    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        plt.figure()
        plt.title('Threshold: ' + str(IPTG_conc))
        for i,distance in enumerate([4.5, 9.0, 13.5]):
            plt.plot(np.array(data_TH[IPTG_conc][distance]).T, colours[i])
    plt.show()