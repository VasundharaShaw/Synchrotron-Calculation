#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test_Synchrotron.py
-------------------
Synchrotron radiation calculation in a uniform magnetic field.

This script:
- Sets up a uniform B-field using CRPropa3
- Computes Stokes Q, U, I along each line of sight in a Healpix map
- Uses analytical synchrotron kernel approximations (F, G)
- Produces Mollweide maps of Q, U, and polarized intensity (PI)

Requirements:
- CRPropa3 installed
- `Imports_python.py` and `physics_constants.py` in the same folder
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import imports_python as imp          # centralised imports (numpy, etc.)
import physics_constants as cons      # physical constants

from crpropa import Vector3d, UniformMagneticField
from numba import vectorize, float64

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import gamma

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
@vectorize([float64(float64)])
def analytical_norm_fun(B):
    """
    Analytical normalization factor for synchrotron emissivity.

    Parameters
    ----------
    B : float
        Magnetic field strength perpendicular to the line of sight [Gauss].

    Returns
    -------
    float
        Normalization constant.
    """
    hbar = cons.h_plank / (2.0 * np.pi)
    return (B / cons.Bcrit) * (cons.m_e / hbar) * (np.sqrt(3.0) * cons.fine_cons) / (4.0 * np.pi)


@vectorize([float64(float64, float64)])
def peak_ener_fun(Ee, B):
    """
    Characteristic synchrotron photon energy (critical energy).

    Ee : float
        Electron energy [eV].
    B : float
        Magnetic field strength [Gauss].

    Returns
    -------
    float
        Photon energy [eV] at synchrotron peak.
    """
    return 1.5 * (Ee / cons.m_e) ** 2 * (B / cons.Bcrit) * cons.m_e


def rotation1(x):
    """Rotation matrix about +Z axis by angle x (radians)."""
    return imp.np.array(((imp.cos(x), imp.sin(x), 0),
                         (-imp.sin(x), imp.cos(x), 0),
                         (0, 0, 1)))


def rotation2(y):
    """Rotation matrix about +Y axis by angle y (radians)."""
    return imp.np.array(((imp.cos(y), 0, -imp.sin(y)),
                         (0, 1, 0),
                         (imp.sin(y), 0, imp.cos(y))))


# ---------------------------------------------------------------------
# Synchrotron kernel approximations (F, G functions)
# ---------------------------------------------------------------------
F1 = imp.pi * 2 ** (5 / 3) / imp.sqrt(3) / imp.gamma(1 / 3)
G1 = F1 / 2
F2 = imp.sqrt(imp.pi / 2)
G2 = F2

# Tuned coefficients
k1, k2, k3 = 1, 2, 3

F_a1_k1, F_a1_k2, F_a1_k3 = -0.97947838884478688, -0.83333239129525072, +0.15541796026816246
F_a2_k1, F_a2_k2, F_a2_k3 = -4.69247165562628882e-2, -0.70055018056462881, +1.038762978419949544e-2
G_a1_k1, G_a1_k2, G_a1_k3 = -1.3746667760953621, +0.44040512552162292, -0.15527012012316799
G_a2_k1, G_a2_k2, G_a2_k3 = -0.33550751062084, 0.0, 0.0

@vectorize([float64(float64)])
def F_delta_1(x): return imp.exp(F_a1_k1 * x ** (1/k1) + F_a1_k2 * x ** (1/k2) + F_a1_k3 * x ** (1/k3))
@vectorize([float64(float64)])
def F_delta_2(x): return 1.0 - imp.exp(F_a2_k1 * x ** (1/k1) + F_a2_k2 * x ** (1/k2) + F_a2_k3 * x ** (1/k3))
@vectorize([float64(float64)])
def G_delta_1(x): return imp.exp(G_a1_k1 * x ** (1/k1) + G_a1_k2 * x ** (1/k2) + G_a1_k3 * x ** (1/k3))
@vectorize([float64(float64)])
def G_delta_2(x): return 1.0 - imp.exp(G_a2_k1 * x ** (1/k1) + G_a2_k2 * x ** (1/k2) + G_a2_k3 * x ** (1/k3))

@vectorize([float64(float64)])
def F_syn(x): return F1 * x ** (1/3) * F_delta_1(x) + F2 * imp.exp(-x) * x ** 0.5 * F_delta_2(x)
@vectorize([float64(float64)])
def G_syn(x): return G1 * x ** (1/3) * G_delta_1(x) + G2 * imp.exp(-x) * x ** 0.5 * G_delta_2(x)


# ---------------------------------------------------------------------
# Healpix sky setup
# ---------------------------------------------------------------------
NSIDE = 32
NPIX = hp.nside2npix(NSIDE)

# Pixel coordinates
theta_gal, phi_gal = hp.pix2ang(NSIDE, np.arange(NPIX))
l = np.degrees(phi_gal)                          # Galactic longitude
b = np.degrees(np.arccos(np.linspace(-1, 1, NPIX))) - 90  # Galactic latitude (approx)

# Frequency band (C-BASS-like, 4.76 GHz)
freq = 4.76  # GHz
Eph_cen = freq * 1e9 * cons.hz_eV


# ---------------------------------------------------------------------
# Main function: Brightness
# ---------------------------------------------------------------------
def Brightness(bpeak, bTur, R, Z, elec_norm):
    """
    Compute Stokes Q, U, I for synchrotron emission with a uniform B field.

    Parameters
    ----------
    bpeak : float
        Magnetic field strength [µG].
    bTur : float
        Turbulent component [µG] (unused in this simplified version).
    R, Z : float
        Radial/vertical scales (unused here).
    elec_norm : float
        Electron density normalization factor.

    Returns
    -------
    tuple
        (Stokes_Q, Stokes_U, I_map, l, b)
    """

    # Uniform magnetic field along z, scaled to CRPropa units
    bField = UniformMagneticField(Vector3d(0, 0, bpeak) * cons.muG)

    N = 10              # number of LoS steps
    dr = 1 * imp.kpc    # step size
    pix_len = len(l)

    # Electron energy grid
    Ee_min, Ee_max, npoints = 1e9, 1e12, 30
    elec_ener = imp.np.logspace(np.log10(Ee_min), np.log10(Ee_max), num=npoints, endpoint=False)
    dlogE = np.log10(elec_ener[1]) - np.log10(elec_ener[0])

    # Outputs
    Stokes_Q, Stokes_U, I_test = np.zeros(pix_len), np.zeros(pix_len), np.zeros(pix_len)

    # Earth position in Galactocentric coords
    pos_earth = Vector3d(-8.5, 0, 0) * imp.kpc

    for i in range(pix_len):
        lon, lat = l[i] * imp.pi / 180., b[i] * imp.pi / 180.
        Q_l, U_l, I_tot_los = 0, 0, 0

        for n in range(1, N):
            r = dr * n

            # Position along LoS
            x = r * imp.cos(lat) * imp.cos(lon)
            y = r * imp.cos(lat) * imp.sin(lon)
            z = r * imp.sin(lat)
            pos = Vector3d(x, y, z) + pos_earth

            # Magnetic field at pos (Vector3d, in Gauss)
            B = bField.getField(pos) / imp.gauss

            # LoS unit vector as Vector3d
            dir_vec = Vector3d(imp.cos(lat) * imp.cos(lon),
                               imp.cos(lat) * imp.sin(lon),
                               imp.sin(lat))

            # Parallel component of B
            B_para_mag = B.dot(dir_vec)
            B_para_vec = dir_vec * B_para_mag

            # Rotations
            B_para_rot1 = imp.np.matmul(rotation1(lon), B_para_vec)
            B_rot1 = imp.np.matmul(rotation1(lon), B)
            B_para_rot2 = imp.np.matmul(rotation2(-lat + imp.pi/2), B_para_rot1)
            B_rot2 = imp.np.matmul(rotation2(-lat + imp.pi/2), B_rot1)

            # Perpendicular component
            B_para_rot_xyz = Vector3d(B_para_rot2[0], B_para_rot2[1], B_para_rot2[2])
            B_rot_xyz = Vector3d(B_rot2[0], B_rot2[1], B_rot2[2])
            B_perp_vec = B_rot_xyz - B_para_rot_xyz
            B_perp_mag = B_perp_vec.getR()

            if B_perp_mag == 0:
                continue

            # Polarization angle
            theta = imp.atan2(B_perp_vec[1], B_perp_vec[0])
            if theta >= imp.np.pi/2: theta -= imp.np.pi
            elif theta < -imp.np.pi/2: theta += imp.np.pi

            # Photon scaling
            Eg_Epeak = Eph_cen / (peak_ener_fun(elec_ener[:], B_perp_mag))
            analytical_norm = analytical_norm_fun(B_perp_mag)

            # Synchrotron emissivities
            J_perp = elec_norm * analytical_norm * (dr*100) / (4*cons.pi) * (F_syn(Eg_Epeak)+G_syn(Eg_Epeak)) * dlogE
            J_para = elec_norm * analytical_norm * (dr*100) / (4*cons.pi) * (F_syn(Eg_Epeak)-G_syn(Eg_Epeak)) * dlogE

            # Integrate along LoS
            I_tot_los += imp.np.sum(J_perp + J_para)
            Q_l += imp.np.sum((J_perp - J_para) * imp.np.cos(2*theta))
            U_l += imp.np.sum((J_perp - J_para) * imp.np.sin(2*theta))

        Stokes_Q[i] = Q_l
        Stokes_U[i] = U_l
        I_test[i] = I_tot_los

    return Stokes_Q, Stokes_U, I_test, l, b




# ---------------------------------------------------------------------
# Run example and plot
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    bpeak, bTur, r_cut, z_max, norm = 2, 0, 0, 0, 1e-13
    Q, U, I_map, l_out, b_out = Brightness(bpeak, bTur, r_cut, z_max, norm)

    # Smooth and plot
    deg = 1.0
    _, Q_s, U_s = hp.ud_grade(hp.smoothing([Q*0, Q, U], np.radians(deg)), NSIDE)
    PI = np.sqrt(Q_s**2 + U_s**2)

    hp.mollview(Q_s, cmap="RdBu", title="Stokes Q (Uniform Bz)", min=-0.04, max=0.04)
    hp.graticule()
    plt.show()

    hp.mollview(U_s, cmap="RdBu", title="Stokes U (Uniform Bz)", min=-0.04, max=0.04)
    hp.graticule()
    plt.show()

    hp.mollview(PI, cmap="jet", title="Polarized Intensity (Uniform Bz)")
    hp.graticule()
    plt.show()

    pol_angs = -np.arctan2(U,Q)/2

    hp.mollview(pol_angs,cmap="hsv",title=r'Polarisation Angles Uniform Bfield $z$ direction')
    hp.graticule()
