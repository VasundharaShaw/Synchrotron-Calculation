#!/usr/bin/env python
# coding: utf-8

# In[1]:



import Imports_python as imp
import physics_constants as cons
from crpropa import *
import time
# import numpy as np
import sys
from numba import jit,vectorize,float64
import healpy as hp
from matplotlib import cm
from scipy.integrate import quad
from scipy.special import kv, gamma
from scipy.integrate import dblquad
# from March_21_Function_Files import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 25})

# In[34]:
direc = '/afs/ifh.de/group/that/work-vs/Para_Scan/Data/'

l    =  imp.np.loadtxt(direc+"Phi.txt")
b    =  -imp.np.loadtxt(direc+"Theta.txt")+90

###########################################################################################################################


log10 = np.log10
@vectorize([float64(float64)])
def analytical_norm_fun(B):
    return (B/cons.Bcrit)* ( cons.m_e / ( cons.h_plank/(2*np.pi) ) ) * ( np.sqrt(3) * cons.fine_cons ) / (4*np.pi)


vec_analytical_norm_fun = imp.np.vectorize(analytical_norm_fun)

@vectorize([float64(float64,float64)])
def peak_ener_fun(Ee,B):
    return  1.5*(Ee/cons.m_e)**2*(B/cons.Bcrit)*cons.m_e

# @vectorize([float64(float64)])
def rotation1(x):
    return imp.np.array(((imp.cos(x),imp.sin(x),0),(-imp.sin(x),imp.cos(x),0),(0,0,1)))
# @vectorize([float64(float64)])
def rotation2(y):
    return imp.np.array(((imp.cos(y),0,-imp.sin(y)),(0,1,0),(imp.sin(y),0,imp.cos(y))))


@vectorize([float64(float64,float64,float64,float64,float64,float64,float64,float64)])
def EdNdE(elec_ener,E_10,norm,dlogE,R_elec,h_r,h_z,z):
    return norm*(elec_ener/E_10)**-2*dlogE *imp.np.exp(-R_elec/(h_r))*((1-imp.tanh(z/(h_z))**2))

    
F1 = imp.pi*2**(5/3)/imp.sqrt(3) / imp.gamma(1/3)
G1 = F1/2
F2 = imp.sqrt(imp.pi/2)
G2 = F2

k1 = 1
k2 = 2
k3 = 3

F_a1_k1 = -0.97947838884478688
F_a1_k2 = -0.83333239129525072
F_a1_k3 =  0.15541796026816246

F_a2_k1 = -4.69247165562628882e-2
F_a2_k2 = -0.70055018056462881
F_a2_k3 =  1.038762978419949544e-2

G_a1_k1 = -1.3746667760953621
G_a1_k2 = +0.44040512552162292
G_a1_k3 = -0.15527012012316799

G_a2_k1 = -0.33550751062084
G_a2_k2 = 0.
G_a2_k3 = 0.


######################################## Bessel functions ###################################

@vectorize([float64(float64)])
def F_delta_1(x):
    return imp.exp(F_a1_k1*x**(1/k1)+F_a1_k2*x**(1/k2)+F_a1_k3*x**(1/k3))
@vectorize([float64(float64)])
def F_delta_2(x):
    return 1-imp.exp(F_a2_k1*x**(1/k1)+F_a2_k2*x**(1/k2)+F_a2_k3*x**(1/k3))
@vectorize([float64(float64)])
def G_delta_1(x):
    return imp.exp(G_a1_k1*x**(1/k1)+G_a1_k2*x**(1/k2)+G_a1_k3*x**(1/k3))
@vectorize([float64(float64)])
def G_delta_2(x):
    return 1-imp.exp(G_a2_k1*x**(1/k1)+G_a2_k2*x**(1/k2)+G_a2_k3*x**(1/k3))

@vectorize([float64(float64)])
def F_syn(x):
    return F1*x**(1./3)*F_delta_1(x) + F2*imp.exp(-x)*x**(1/2.)*F_delta_2(x)

@vectorize([float64(float64)])
def G_syn(x):
    return G1*x**(1/3)*G_delta_1(x) + G2*imp.exp(-x)*x**(1/2)*G_delta_2(x)

#########################################################################################


pix_len = len(l)
## 28.4 GHz
freq = 4.76
Eph_min = (freq-1.5)*1e9*cons.hz_eV
Eph_cen = freq*1e9*cons.hz_eV
Eph_max = (freq+1.5)*1e9*cons.hz_eV

eV_erg = 1.6e-12
(Eph_max),(Eph_min)

E_phs = np.logspace(np.log10(Eph_min),np.log10(Eph_max),7,endpoint = False)
#print('Egs = ',E_phs)

d_log_step = np.log10(E_phs[1]/E_phs[0]) 
#print('dlog_10 (step size) = ',d_log_step)
randomSeed = 10
zmin = 0.1*kpc

nside_out =  32# Choose appropriate nside value for your map


# Get the number of pixels
npix = hp.nside2npix(nside_out)

# Get the declination of each pixel
theta_gal_64, phi_gal_64 = hp.pix2ang(nside_out, np.arange(npix))
l = np.degrees((phi_gal_64) )
b = np.degrees(np.arccos(np.linspace(-1,1,npix)))-90



def Brightness(bpeak,bTur,R,Z,elec_norm):
    bField  = JF12() ## Enter Magnetic field object here. Using CRPropa's JF12Field class as an example

    seed = 691342
    # bField.randomStriated(seed)
    # bField.randomTurbulent(seed)   

    # N = 320
    # dr = 0.1*imp.kpc
    N = 32
    dr = 1*imp.kpc
    pix_len = len(l)


    Ee_min  = 1e9
    Ee_max  = 1e12
    num_decades = np.log(Ee_max/Ee_min)


    eV_erg = 1.6e-12

    bin_width = imp.np.exp(1.)
    log10     = imp.log10
    npoints = 30
    Ee_10 = 10e9 
    elec_ener =  imp.np.logspace(np.log10(Ee_min), np.log10(Ee_max), num=npoints, endpoint=False)

    dlogE =  np.log10(elec_ener[1])-np.log10(elec_ener[0])


    # ######################################## synchrotron calculations ##################################
    counter = 0
    P_test,I_test,h_r,h_z     = imp.np.zeros([pix_len],dtype=float),imp.np.zeros([pix_len],dtype=float),R*imp.kpc,Z*imp.kpc
    pixel = 0
    P_ang = np.zeros(pix_len)
    R_elec_val,Ne_val = np.zeros([pix_len,N]),np.zeros([pix_len,N])
    Stokes_Q,Stokes_U = np.zeros([pix_len]),np.zeros([pix_len])
    for i in range (0,pix_len): 

        theta,R_elec,Q_l,U_l,Q_l_sum,U_l_sum = 0,0,0,0,0,0
        I_tot_los = 0
        time_los_0 = time.time()
        for n in range(1,N): ##distance going from 0 to  20 imp.kpc: 

            x0,y0,z0 = -8.5,0,0

            pos_earth = Vector3d(x0,y0,z0) *imp.kpc

            r  = dr*(n) 
            # if((0<=l[i]<=110 or 250<=l[i]<=360)):## 
            # if((0<=l[i]<=90 or 270<=l[i]<=360)):## 
            # if((0<=l[i]<=160 or 200<=l[i]<=360)):## 

            # if((0<=l[i]<=90 or 270<=l[i]<=360) and (15<=b[i] or b[i]<=-15)):## 
            # if((15<=b[i] or b[i]<=-15)):## 

            if((0<=l[i]<=180 or 180<=l[i]<=360)):## 

                lon,lat = l[i]*imp.np.pi/180.,b[i]*imp.np.pi/180.
                #  
                x  = r * imp.np.cos(lat) * imp.np.cos(lon)
                y  = r * imp.np.cos(lat) * imp.np.sin(lon)
                z  = r * imp.np.sin(lat)


                pos  = Vector3d(x, y, z) + pos_earth ## (Corodinate system is centered at GC now and r starts at earth)

                R_elec =imp.np.sqrt(pos[0]**2+pos[1]**2)   

                mag_pos = pos.getR() 

                # B =  bField.getField(pos) / imp.gauss +   (bTu*bField_tur[i,n,:])
                B =  bField.getField(pos) / imp.gauss 

                #print()

                Bx =  B[0]
                By =  B[1]
                Bz =  B[2]
                dir_x,dir_y,dir_z  = imp.cos(lat)*imp.cos(lon),imp.cos(lat)*imp.sin(lon),imp.sin(lat)

                B_para_mag = ((Bx*dir_x+By*dir_y+Bz*dir_z))
                B_para_vec = Vector3d(dir_x,dir_y,dir_z)*B_para_mag

        ############################################# Rotation along  anti clockwise #######################################################    

                B_para_rot1 = imp.np.matmul(rotation1(lon),B_para_vec)
                B_rot1      = imp.np.matmul(rotation1(lon),B)

        ############################################### Rotation along Y #######################################################    


                B_para_rot2 = imp.np.matmul(rotation2(-lat+imp.np.pi/2),B_para_rot1)
                B_rot2      = imp.np.matmul(rotation2(-lat+imp.np.pi/2),B_rot1)

        ##################################################################

                B_para_rot_xyz = Vector3d(B_para_rot2[0],B_para_rot2[1],B_para_rot2[2])
                B_para_rot_mag = B_para_rot_xyz.getR()
              
                B_rot_xyz = Vector3d(B_rot2[0],B_rot2[1],B_rot2[2])    

                B_perp_vec = B_rot_xyz-B_para_rot_xyz
        ############################################################### Polarisation ##########################################################            

                B_perp_mag = B_perp_vec.getR()

                B_perp_dir = B_perp_vec/(B_perp_mag) 
                J_para_x_dir = B_perp_dir[0]
                J_para_y_dir = B_perp_dir[1]

                theta = imp.atan2(B_perp_vec[1],B_perp_vec[0])  

                dV = dr*100.*(r*100.)**2*cons.d_omega
                dA = (r*100.)**2*cons.d_omega

                if ((theta) >= imp.np.pi/2):
                    theta = theta - imp.np.pi
                elif((theta) < -imp.np.pi/2):
                    theta = theta + imp.np.pi
                else:
                    theta = theta
                # for k in range(0,7):
                if (B_perp_mag != 0):

                    Eg_Epeak = E_phs[:]/(peak_ener_fun(elec_ener[:,None],B_perp_mag))



                analytical_norm  = analytical_norm_fun(B_perp_mag) 

                norm  = elec_norm
                
                #print(norm)

                J_perp = EdNdE(elec_ener[:,None],Ee_10,norm,dlogE,R_elec,h_r,h_z,z)*analytical_norm*(dr*100)/(4*cons.pi)*(F_syn(Eg_Epeak)+G_syn(Eg_Epeak))*d_log_step
                J_para = EdNdE(elec_ener[:,None],Ee_10,norm,dlogE,R_elec,h_r,h_z,z)*analytical_norm*(dr*100)/(4*cons.pi)*(F_syn(Eg_Epeak)-G_syn(Eg_Epeak))*d_log_step


                # #print(J_perp)

                I_tot_los += J_perp+J_para
                Q_l += (J_perp - J_para)*imp.np.cos(2*theta)
                U_l += (J_perp - J_para)*imp.np.sin(2*theta)

                #print()
                #print(Q_l)
                #print()
            
        # sys.exit()

        Q_l_sum = imp.np.sum(Q_l)
        U_l_sum = imp.np.sum(U_l)
        Stokes_Q[i] = Q_l_sum
        Stokes_U[i] = U_l_sum

        P_test[i] = imp.np.sqrt(Q_l_sum**2+U_l_sum**2)
        I_test[i] = np.sum(I_tot_los)

        #print()
        #print('i = ', i,'  n =  ', n , '  ',Stokes_Q[i],'  ', Stokes_U[i], '   ',I_test[i])

    return Stokes_Q,Stokes_U,I_test


%%time
bpeak,bTur,r_cut,z_max,norm = 0,0,5,6,1e-13
# bpeak,bTur,r_cut,z_max,norm = 9.1,2.33,19,6,1e-13

P_syn = Brightness(bpeak,bTur,r_cut,z_max,norm)

#!/usr/bin/env python
# coding: utf-8

# === Module Imports ===
import Imports_python as imp                        # Custom module with preloaded packages/constants
import physics_constants as cons                   # Custom module defining physical constants (e.g., Bcrit, m_e, h_plank, etc.)
from crpropa import *                              # CRPropa3 for magnetic field and vector operations
import time
import numpy as np
import sys
from numba import jit, vectorize, float64          # JIT compilation and vectorization for performance
import healpy as hp                                # HEALPix functions for sky pixelation
from matplotlib import cm
from scipy.integrate import quad, dblquad          # Numerical integration
from scipy.special import kv, gamma                # Special functions for synchrotron calculations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 25})      # Increase default font size for plots

# === Load Sky Coordinates (Galactic) ===
direc = '/afs/ifh.de/group/that/work-vs/Para_Scan/Data/'

l = imp.np.loadtxt(direc + "Phi.txt")              # Galactic longitude in degrees
b = -imp.np.loadtxt(direc + "Theta.txt") + 90      # Convert theta (polar angle) to Galactic latitude

###########################################################################################################################

# === Convenience function ===
log10 = np.log10

# === Analytical normalization function for synchrotron emissivity ===
@vectorize([float64(float64)])
def analytical_norm_fun(B):
    return (B / cons.Bcrit) * (cons.m_e / (cons.h_plank / (2 * np.pi))) * (np.sqrt(3) * cons.fine_cons) / (4 * np.pi)

vec_analytical_norm_fun = imp.np.vectorize(analytical_norm_fun)  # Vectorized version using np.vectorize

# === Characteristic photon energy (peak) for given electron energy and magnetic field ===
@vectorize([float64(float64, float64)])
def peak_ener_fun(Ee, B):
    return 1.5 * (Ee / cons.m_e) ** 2 * (B / cons.Bcrit) * cons.m_e

# === Rotation matrices (used for aligning magnetic field vectors) ===
def rotation1(x):
    return imp.np.array((
        (imp.cos(x), imp.sin(x), 0),
        (-imp.sin(x), imp.cos(x), 0),
        (0, 0, 1)
    ))

def rotation2(y):
    return imp.np.array((
        (imp.cos(y), 0, -imp.sin(y)),
        (0, 1, 0),
        (imp.sin(y), 0, imp.cos(y))
    ))

# === Electron distribution function EdNdE with exponential falloff and tanh(z) suppression ===
@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64)])
def EdNdE(elec_ener, E_10, norm, dlogE, R_elec, h_r, h_z, z):
    return norm * (elec_ener / E_10) ** -2 * dlogE * imp.np.exp(-R_elec / (h_r)) * (1 - imp.tanh(z / (h_z)) ** 2)

# === Synchrotron fit constants from literature or empirical fitting ===
F1 = imp.pi * 2 ** (5 / 3) / imp.sqrt(3) / imp.gamma(1 / 3)
G1 = F1 / 2
F2 = imp.sqrt(imp.pi / 2)
G2 = F2

# === Exponents used for Bessel-function-like fitting ===
k1 = 1
k2 = 2
k3 = 3

# === Fit coefficients for approximating Bessel functions in F_syn and G_syn ===
# F_delta and G_delta describe different polarization components of synchrotron emission
F_a1_k1, F_a1_k2, F_a1_k3 = -0.97947838884478688, -0.83333239129525072, 0.15541796026816246
F_a2_k1, F_a2_k2, F_a2_k3 = -4.69247165562628882e-2, -0.70055018056462881, 1.038762978419949544e-2
G_a1_k1, G_a1_k2, G_a1_k3 = -1.3746667760953621, +0.44040512552162292, -0.15527012012316799
G_a2_k1, G_a2_k2, G_a2_k3 = -0.33550751062084, 0., 0.

######################################## Bessel-like fitting functions for synchrotron emission ###################################

# === These approximate the synchrotron kernel using exponential fits ===

@vectorize([float64(float64)])
def F_delta_1(x):
    return imp.exp(F_a1_k1 * x ** (1 / k1) + F_a1_k2 * x ** (1 / k2) + F_a1_k3 * x ** (1 / k3))

@vectorize([float64(float64)])
def F_delta_2(x):
    return 1 - imp.exp(F_a2_k1 * x ** (1 / k1) + F_a2_k2 * x ** (1 / k2) + F_a2_k3 * x ** (1 / k3))

@vectorize([float64(float64)])
def G_delta_1(x):
    return imp.exp(G_a1_k1 * x ** (1 / k1) + G_a1_k2 * x ** (1 / k2) + G_a1_k3 * x ** (1 / k3))

@vectorize([float64(float64)])
def G_delta_2(x):
    return 1 - imp.exp(G_a2_k1 * x ** (1 / k1) + G_a2_k2 * x ** (1 / k2) + G_a2_k3 * x ** (1 / k3))

# === Total synchrotron emission kernel for perpendicular (F) and parallel (G) components ===

@vectorize([float64(float64)])
def F_syn(x):
    return F1 * x ** (1. / 3) * F_delta_1(x) + F2 * imp.exp(-x) * x ** (1 / 2.) * F_delta_2(x)

@vectorize([float64(float64)])
def G_syn(x):
    return G1 * x ** (1 / 3) * G_delta_1(x) + G2 * imp.exp(-x) * x ** (1 / 2) * G_delta_2(x)
#########################################################################################

# === Setup for synchrotron frequency range ===
pix_len = len(l)                    # Number of pixels

# Central observing frequency in GHz (28.4 GHz used here)
freq = 4.76

# Convert frequency ±1.5 GHz to photon energy range in eV
Eph_min = (freq - 1.5) * 1e9 * cons.hz_eV
Eph_cen = freq * 1e9 * cons.hz_eV
Eph_max = (freq + 1.5) * 1e9 * cons.hz_eV

eV_erg = 1.6e-12                    # Conversion factor from eV to erg

# Define photon energy bins on log scale
E_phs = np.logspace(np.log10(Eph_min), np.log10(Eph_max), 7, endpoint=False)

# Step size in log10 space between energy bins
d_log_step = np.log10(E_phs[1] / E_phs[0])

# === Simulation configuration ===
randomSeed = 10
zmin = 0.1 * kpc                   # Minimum z-distance

# === HEALPix map configuration ===
nside_out = 32                     # Output NSIDE (defines map resolution)
npix = hp.nside2npix(nside_out)   # Total number of pixels

# Get angular coordinates for each HEALPix pixel (in radians)
theta_gal_32, phi_gal_32 = hp.pix2ang(nside_out, np.arange(npix))

# Convert to degrees for galactic longitude and latitude
l = np.degrees(phi_gal_32)
b = np.degrees(np.arccos(np.linspace(-1, 1, npix))) - 90  # Approximate latitude array

# === Main function to compute synchrotron brightness and polarization ===
# Returns Stokes Q, U, and total intensity I maps
def Brightness(bpeak, bTur, R, Z, elec_norm):
    bField = JF12()                          # CRPropa's JF12 Galactic magnetic field model

    seed = 691342                            # Optional seed for striated/turbulent components (disabled here)

    N = 32                                   # Number of integration steps along line-of-sight
    dr = 1 * imp.kpc                         # Step size in kpc

    pix_len = len(l)                         # Number of sky pixels

    # === Electron energy configuration ===
    Ee_min = 1e9                             # 1 GeV minimum
    Ee_max = 1e12                            # 1 TeV maximum
    num_decades = np.log(Ee_max / Ee_min)

    bin_width = imp.np.exp(1.)
    npoints = 30                             # Number of energy bins
    Ee_10 = 10e9                             # Reference energy (10 GeV)

    # Electron energy bins (log scale)
    elec_ener = imp.np.logspace(np.log10(Ee_min), np.log10(Ee_max), num=npoints, endpoint=False)
    dlogE = np.log10(elec_ener[1]) - np.log10(elec_ener[0])

    # === Initialize output arrays ===
    P_test, I_test = imp.np.zeros([pix_len], dtype=float), imp.np.zeros([pix_len], dtype=float)
    h_r, h_z = R * imp.kpc, Z * imp.kpc                          # Scale heights
    P_ang = np.zeros(pix_len)
    R_elec_val, Ne_val = np.zeros([pix_len, N]), np.zeros([pix_len, N])
    Stokes_Q, Stokes_U = np.zeros([pix_len]), np.zeros([pix_len])  # Stokes parameters Q and U

    # === Loop over all sky pixels ===
    for i in range(pix_len):
        # Initialize variables for each line-of-sight
        theta, R_elec, Q_l, U_l, Q_l_sum, U_l_sum = 0, 0, 0, 0, 0, 0
        I_tot_los = 0                          # Total intensity for this pixel

        for n in range(1, N):                 # Integrate along the line-of-sight
            x0, y0, z0 = -8.5, 0, 0           # Earth’s position in kpc (Galactocentric)
            pos_earth = Vector3d(x0, y0, z0) * imp.kpc

            r = dr * n                        # Distance from observer in kpc

            # Restrict calculation to selected Galactic longitude regions
            if (0 <= l[i] <= 180 or 180 <= l[i] <= 360):
                # Convert angles to radians
                lon, lat = l[i] * imp.np.pi / 180., b[i] * imp.np.pi / 180.

                # Convert spherical to Cartesian (GC-centered)
                x = r * imp.np.cos(lat) * imp.np.cos(lon)
                y = r * imp.np.cos(lat) * imp.np.sin(lon)
                z = r * imp.np.sin(lat)

                # Position vector from GC
                pos = Vector3d(x, y, z) + pos_earth

                R_elec = imp.np.sqrt(pos[0]**2 + pos[1]**2)
                mag_pos = pos.getR()

                # Get magnetic field at this point
                B = bField.getField(pos) / imp.gauss

                # Decompose B field
                Bx, By, Bz = B[0], B[1], B[2]
                dir_x, dir_y, dir_z = imp.cos(lat) * imp.cos(lon), imp.cos(lat) * imp.sin(lon), imp.sin(lat)

                # Parallel component of B
                B_para_mag = Bx * dir_x + By * dir_y + Bz * dir_z
                B_para_vec = Vector3d(dir_x, dir_y, dir_z) * B_para_mag

                # === Rotate B field into LOS-aligned coordinate frame ===
                B_para_rot1 = imp.np.matmul(rotation1(lon), B_para_vec)
                B_rot1 = imp.np.matmul(rotation1(lon), B)

                B_para_rot2 = imp.np.matmul(rotation2(-lat + imp.np.pi / 2), B_para_rot1)
                B_rot2 = imp.np.matmul(rotation2(-lat + imp.np.pi / 2), B_rot1)

                # Final rotated vectors
                B_para_rot_xyz = Vector3d(*B_para_rot2)
                B_rot_xyz = Vector3d(*B_rot2)

                # Perpendicular magnetic field component (in image plane)
                B_perp_vec = B_rot_xyz - B_para_rot_xyz

                # === Magnitude and direction of B_perpendicular ===
                B_perp_mag = B_perp_vec.getR()

                # Normalize the perpendicular direction vector
                B_perp_dir = B_perp_vec / B_perp_mag
                J_para_x_dir = B_perp_dir[0]
                J_para_y_dir = B_perp_dir[1]

                # Polarization angle in the plane of the sky
                theta = imp.atan2(B_perp_vec[1], B_perp_vec[0])

                # === Volume element and area for emission ===
                dV = dr * 100. * (r * 100.) ** 2 * cons.d_omega  # in cm³
                dA = (r * 100.) ** 2 * cons.d_omega              # area in cm²

                # Normalize polarization angle to [-π/2, π/2]
                if theta >= imp.np.pi / 2:
                    theta -= imp.np.pi
                elif theta < -imp.np.pi / 2:
                    theta += imp.np.pi

                # === Synchrotron emission calculation (only if B_perp is non-zero) ===
                if B_perp_mag != 0:
                    # Dimensionless ratio of photon energy to critical energy
                    Eg_Epeak = E_phs[:] / peak_ener_fun(elec_ener[:, None], B_perp_mag)

                    # Synchrotron normalization factor
                    analytical_norm = analytical_norm_fun(B_perp_mag)

                    norm = elec_norm  # Electron density normalization

                    # === Compute polarized synchrotron emissivities (Stokes I, Q, U) ===
                    J_perp = (
                        EdNdE(elec_ener[:, None], Ee_10, norm, dlogE, R_elec, h_r, h_z, z)
                        * analytical_norm
                        * (dr * 100) / (4 * cons.pi)
                        * (F_syn(Eg_Epeak) + G_syn(Eg_Epeak))
                        * d_log_step
                    )

                    J_para = (
                        EdNdE(elec_ener[:, None], Ee_10, norm, dlogE, R_elec, h_r, h_z, z)
                        * analytical_norm
                        * (dr * 100) / (4 * cons.pi)
                        * (F_syn(Eg_Epeak) - G_syn(Eg_Epeak))
                        * d_log_step
                    )

                    # === Add intensity and polarization contributions ===
                    I_tot_los += J_perp + J_para
                    Q_l += (J_perp - J_para) * imp.np.cos(2 * theta)
                    U_l += (J_perp - J_para) * imp.np.sin(2 * theta)

        # === Final Stokes parameters for this sky pixel ===
        Q_l_sum = imp.np.sum(Q_l)
        U_l_sum = imp.np.sum(U_l)
        Stokes_Q[i] = Q_l_sum
        Stokes_U[i] = U_l_sum

        # Total polarized intensity
        P_test[i] = imp.np.sqrt(Q_l_sum**2 + U_l_sum**2)

        # Total intensity (sum over all lines of sight)
        I_test[i] = np.sum(I_tot_los)

    return Stokes_Q, Stokes_U, I_test

# === Measure execution time of brightness computation ===
%%time

# Input parameters:
# bpeak, bTur: placeholders for additional magnetic field components (not used here)
# r_cut: radial scale height of electron distribution (in kpc)
# z_max: vertical scale height of electron distribution (in kpc)
# norm: normalization constant for electron distribution

bpeak, bTur, r_cut, z_max, norm = 0, 0, 5, 6, 1e-13

# === Call the main synchrotron brightness function ===
P_syn = Brightness(bpeak, bTur, r_cut, z_max, norm)

# P_syn contains:
#   - Stokes_Q: Linear polarization component Q
#   - Stokes_U: Linear polarization component U
#   - I_test:   Total intensity map (Stokes I)
