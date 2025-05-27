#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:41:06 2025

@author: ljnolan

Additional figures for the FABULOVS Project
"""

from affogato import getBkgrd, percentile_cut

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

def cut_data(filename, position, size, ext=1):
   with fits.open(filename) as hdul:
      wcs = WCS(hdul[ext].header)
      cutout = Cutout2D(hdul[ext].data, position=position, size=size, wcs=wcs)
   return cutout.data


def convert(c, side, control):
   path = 'galfit_automate'
   if not control:
      catalog = Table.read('HST_resampled/catalog.csv', format='csv')
      name = 'ie47%.2i' % c
      file = 'HST_resampled/%s_drz.fits' % name
      frame = 'icrs'
      c-=1
      pos = SkyCoord(ra=catalog['ra'][c]*u.degree, 
                     dec=catalog['dec'][c]*u.degree, frame=frame)
   else:
      catalog = Table.read('Control_Sample/controlsample.csv', 
                           format='csv')
      name = 'control%.3i' % c
      file = 'Control_Sample/sample/%s_drz.fits' % name
      frame = 'icrs'
      wherecat = catalog[catalog['myID'] == c]
      pos = SkyCoord(ra=wherecat['ra']*u.degree, 
                     dec=wherecat['dec']*u.degree, frame=frame)
   size = [side, side]
   return file, pos, size


def pop_fig(ax, c, control):
   ax.set_axis_off()
   ax.grid()
   scale = 0.06
   
   # Set image size
   side = 400
   if not control:
      if c in (4,6):
        side += 200
   
   # Data manipulation and scaling
   file, pos, size = convert(c, side, control)
   data = cut_data(file, pos, size)
   llim, ulim = np.percentile(data, [1, 99])
   bkgrd = getBkgrd(file, 1)
   norm = ImageNormalize(stretch=LogStretch(), vmin=bkgrd, vmax=ulim)
   datat = percentile_cut(data, 1, 99)
   
   # Scalebar
   corn = len(data) / 20
   pts = [[corn, corn+(5/scale)], [corn, corn]]
   ax.plot(pts[0], pts[1], 'w-', linewidth=4)
   ax.text(corn+(2.4/scale), corn * (6.5/5), '5"', color='w', 
           fontweight='bold', fontsize='large')
   
   # Appearance
   origin = 'lower'
   cmap = 'viridis'
   interpolation = 'nearest'
   
   # Plot
   ax.imshow(datat, norm=norm, origin=origin, cmap=cmap, 
             interpolation=interpolation)
   return


def sample_grid(control):
   if control:
      sam = [2, 3, 14, 25, 40, 44, 48, 85]
   else:
      sam = list(range(9)[1:])
   fig, axs = plt.subplots(2,4, figsize=(24,12))
   for n in range(8):
      pop_fig(axs[n//4, n%4], sam[n], control)
   fig.tight_layout()
   plt.show()
   plt.clf()


sample_grid(control=False)
sample_grid(control=True)
