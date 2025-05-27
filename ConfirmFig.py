# -*- coding: utf-8 -*-
"""

Liam Nolan
Figure for confirmation of size-brightness scaling
11/12/2024

"""

from affogato import from_latex, plotutils
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.table import join
plt.style.use(astropy_mpl_style)


def abs_mag(app_mag, z): # could add cosmology='WMAP9'
   # exec{f'from astropy.cosmology import {cosmology} as cosmo'}
   d = cosmo.comoving_distance(z).to(u.pc).value
   return app_mag - (5 * (np.log10(d) - 1))


def reff_mag(model):
   z = model['z'].data
   reff_as = model[r'$R_{eff}$'].data * 0.06 * u.arcsec
   reff = reff_as * cosmo.kpc_proper_per_arcmin(z)
   reff = reff.to(u.kpc)
   mag = abs_mag(model['Mag'].data, z)
   return reff, mag


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CBcc = CB_color_cycle

backupdir = 'backup'
model = from_latex(backupdir, 'model.tab')
stats = from_latex(backupdir, 'stats.tab')
coords = from_latex('Control_Sample', 'coords.tab')

model2 = model[model[r'Fit $\chi^{2}/\nu$'] > 0]
model2 = join(model2, coords, keys='Name')

models, modelc = model2[model2['kind']==0], model2[model2['kind']==1]

reffs, mags = reff_mag(models)
reffc, magc = reff_mag(modelc)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(reffs, mags, c=CBcc[0], marker='*', label='SoI')
ax.scatter(reffc, magc, c=CBcc[1], marker='o', label='Control')
ax.set_xlabel(r'$R_{eff}$ [kpc]')
ax.set_ylabel('Abs Mag')
ax.grid()
plt.gca().invert_yaxis()
ax.legend()
