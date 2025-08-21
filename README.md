# Synchrotron_Radiation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  
[![CRPropa3](https://img.shields.io/badge/CRPropa3-3.2-green.svg)](https://crpropa.github.io/CRPropa3/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides tools for **calculating synchrotron radiation** from charged particles in arbitrary Galactic Magnetic Field (GMF) models.  
It builds upon **CRPropa3** for particle propagation and uses analytical synchrotron formulations for emission spectra.  

📄 Please see the related paper for reference:  
[Shaw et al., *Synchrotron emission in Galactic Magnetic Fields*, MNRAS 517, 2534 (2023)](https://academic.oup.com/mnras/article/517/2/2534/6731784).

---

## 🚀 Features
- Particle propagation and trajectory integration with **CRPropa3**  
- Synchrotron emission spectra from relativistic electrons in arbitrary GMFs  
- Numerical integration routines (SciPy, Numba acceleration)  
- Healpy-based sky projections  
- Modular constants (`physics_constants.py`) and reusable imports (`Imports_python.py`)  
- Test scripts for example calculations (`Test_Synchrotron.py`)  

---

## 📦 Requirements

This project requires **Python 3.10+** and the following libraries:

| Package      | Recommended Version |
|--------------|----------------------|
| [CRPropa3](https://crpropa.github.io/CRPropa3/pages/Installation.html) | 3.2+ |
| numpy        | 1.23+ |
| scipy        | 1.10+ |
| astropy      | 5.3+ |
| healpy       | 1.16+ |
| matplotlib   | 3.7+ |
| numba        | 0.57+ |
| emcee        | 3.1+ |
| cmcrameri    | 1.7+ |
| colorcet     | 3.0+ |
| sympy        | 1.12+ |
| tqdm         | 4.65+ |

> ⚠️ Ensure that your `numpy` and other libraries are compatible with the installed CRPropa3 version.  
> Using a virtual environment (`conda` or `venv`) is **highly recommended**.

---

## ⚙️ Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/Synchrotron_Radiation.git
cd Synchrotron_Radiation

conda create -n synchrotron python=3.10
conda activate synchrotron

pip install numpy==1.23.5 scipy==1.10 astropy==5.3 healpy==1.16.5 matplotlib==3.7 \
            numba==0.57 emcee==3.1 cmcrameri==1.7 colorcet==3.0 sympy==1.12 tqdm==4.65
