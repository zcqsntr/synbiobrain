import math
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


V_AHL = 15 * 10**(-6) #L
C_AHL = 0.01 * 10**(-3) #M
Pt = V_AHL * C_AHL #mol
print(Pt)
T = 0.00314 #m
D = 5*10**(-10) #m^2/s
t = 16 #hours
t = t*60**2 #seconds
print(t)



def p_density(x):
    return 2*1/((4*np.pi * D * t)**(1/2)) * np.exp(-x**2/(4*D*t)) # mulpily by two for each side of the curve

def integrate_p_density(x0, x1):
    prob, err = quad(p_density, x0, x1)
    print(err)
    return prob

def get_N_particles(x0, x1):
    prob = integrate_p_density(x0,x1)
    print('N_particles: ', prob*Pt)
    return prob * Pt

def get_volume(x0, x1):
    vol = math.pi * (x1**2 - x0**2) * T
    print('vol: ', vol)
    return vol

def get_conc_uM(x0, x1):
    mol_per_m3 = get_N_particles(x0, x1)/get_volume(x0, x1)
    return mol_per_m3 * 1000










print('Concentration of AHL in first half cm: (uM)', get_conc_uM(0.0, 0.005))
print('Concentration of AHL in last half cm: (uM)', get_conc_uM(0.04, 0.045))
print('Concentration of AHL that kills cells: (uM)', get_conc_uM(0.0249, 0.0251))



print('Concentration of AHL in first half cm: (uM)', get_conc_uM(0.0, 0.005))
