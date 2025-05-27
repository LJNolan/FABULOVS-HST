# -*- coding: utf-8 -*-
"""

Liam Nolan
Temporary working file
9/23/2022

"""

from affogato import get_flags, input_to_guess, to_latex, from_latex, \
                       get_header_param, dataPull
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import LogStretch, PercentileInterval, \
                                  astropy_mpl_style
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.utils.data import get_pkg_data_filename
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from photutils.segmentation import detect_threshold, detect_sources, \
                                   deblend_sources, SourceCatalog
from photutils.utils import circular_footprint
plt.style.use(astropy_mpl_style)


def trimPercentile(data, n):
# =============================================================================
# Trims given array, setting values below the nth percentile and above the
# 100 - nth percentile to their respective limits.
# =============================================================================
   plower = np.nanpercentile(data, n)
   phigher = np.nanpercentile(data, 100-n)
   data2 = np.copy(data)
   data2[data2 < plower] = plower
   data2[data2 > phigher] = phigher
   return data2


def showMe(data, title=None, filename=None):
   fig, ax = plt.subplots(figsize=(6,6))
   ax.set_axis_off()
   ax.grid()
   norm = ImageNormalize(stretch=LogStretch())
   origin = 'lower'
   cmap = 'Greys_r'
   interpolation = 'nearest'
   if title is None:
      filler = 1
   else:
      ax.set_title(title)
   im = ax.imshow(data, origin=origin, cmap=cmap, 
             interpolation=interpolation)
   
   divider = make_axes_locatable(ax)
   cax = divider.append_axes('bottom', size='4%', pad=0.02)
   cbar = fig.colorbar(im, cax=cax,orientation='horizontal')
   cbar.ax.tick_params(labelsize=14, width=1.5)
   
   if filename is None:
      plt.show()
   else:
      plt.savefig('New_Driz/%s' % filename, bbox_inches='tight')
   plt.close()
   return


def peeble(c, control=False):
   if control:
      name = 'control%.3i' % c
      oldfile = 'Shreya_Control_Sample/sample/%s_drz.fits' % name
   else:
      name = 'ie47%.2i' % c
      oldfile = 'HST_resampled/%s_drz.fits' % name
   filename = '%s_compare.png' % name
   newfile = 'New_Driz/%s_drz.fits' % name

   olddata = dataPull(oldfile, ext=1)
   newdata = dataPull(newfile, ext=1)

   transform = PercentileInterval(99.)

   olddatat = trimPercentile(olddata, 1)
   newdatat = trimPercentile(newdata, 1)

   bigLad = np.nanmax(newdatat - olddatat)
   smallLad = np.nanmin(newdatat - olddatat)
   
   title = '%s %i %i' % (name, bigLad, smallLad)
   
   showMe(newdatat - olddatat, title=title, filename=filename)
   return
   

control = True
if control:
   dudes = [2, 3, 14, 25, 40, 44, 48, 85]
else:
   dudes = range(1, 9)

for c in dudes:
   peeble(c, control)
