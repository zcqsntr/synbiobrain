from load_spot_data import *
from load_const_data import *
import math

def hill(conc, n, kd, min, max):
    return min + (max-min)*(conc**n/(kd**n + conc**n))

def diffusion_eq(r, t, p, D):
    # diffusion of instantaneous point source
    # concentration at radius r and ttime t of p amount of molcule added with diffusion coeff D

    return p/(4*math.pi*D*t) * math.exp(-r**2/(4*D*t))

# IPTG diffusion in water 6.5 × 10−6 cm2 s−1
    # 8.7×10−10 m2·s−1 at 30◦C
if __name__ == '__main__':
    n_points = 67
    filepath_BP = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201124_IPTGsendersZBD_img_data_summary.csv'
    spot_data_BP = load_spot_data(filepath_BP, n_points)

    n_points = 62
    filepath_TH = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201201_IPTGsendersZG_img_data_summary.csv'
    spot_data_TH = load_spot_data(filepath_TH, n_points)

    filepath = '/home/neythen/Desktop/Projects/synbiobrain/IPTG_characterisation/data/201202_IPTGagar_img_data_summary.csv'
    n_points = 64

    const_data_TH, const_data_BP = load_const_data(filepath, n_points)


    # first fit a sigmoid to the threshold data


    D = 8.7e-10  # m^2/sec
    D *= 6e7  # mm^2/min

    for IPTG_conc in [0., 5., 10., 50., 100., 500.]:
        p = IPTG_conc * 1e-6 # 1 uL of solution, this is in m mols





