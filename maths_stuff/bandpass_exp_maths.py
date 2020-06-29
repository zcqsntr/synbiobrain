import math
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.integrate import dblquad

V_AHL = 15 * 10**(-6) #L
C_AHL = 0.01 * 10**(-3) #M
Pt = V_AHL * C_AHL #mol
print(Pt)
T = 0.00314 #m
D = 5*10**(-10) #m^2/s
t = 16 #hours
t = t*60**2 #seconds
print(t)



def p_density(x, N0 = 1):
    return N0/(4*np.pi * D * t) * np.exp(-x**2/(4*D*t)) # mulpily by two for each side of the curve


def p_density(x, t_prime, N0 = 1):
    return N0/(4*np.pi * D * (t- t_prime)) * np.exp(-x**2/(4*D*(t-t_prime))) # mulpily by two for each side of the curve


def integrate_p_density(x0, x1):
    prob, err = quad(p_density, x0, x1)
    print(err)
    return prob

def integrate_p_density(x0, x1, t0, t1):
    prob, err = dblquad(p_density, t0, t1, x0, x1)
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


t = 10
print('asdfsdaf', integrate_p_density(0, 999999, 0, 10))