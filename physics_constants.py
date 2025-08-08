import numpy as np
import Imports_python as imp

h_plank             =  4.136E-15              # eV.s
m_e                 =  0.511e6                # electron mass in eV
Bcrit               =  4.414e13                # in Gauss neutron star
alpha               =  3.0
# d_cos_theta         =  2./180.
# d_phi               =  2.*imp.pi/180.
fine_cons           =  1./137.
c                   =  3*10**10           # CGS speed of light
R_el                =  12.000
Z_el                =  7.000
hz_eV               = 4.135665538536e-15
d_omega             = 4*imp.pi/12288
pi                  = imp.pi
kpc                 = 1*imp.kpc
kpc_pc              = 1000
log10               = imp.log10
cos                 = imp.cos
sin                 = imp.sin
pc                  = 1*imp.pc
muG                 = 1*imp.muG
gauss               = 1*imp.gauss

######## Bessel functions ########################
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
