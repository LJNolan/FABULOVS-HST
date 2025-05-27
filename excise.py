#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:17:33 2023

@author: ljnolan

This code takes ie4709 and ie4710, our dedicated PSF exposures, excises from 
4710 a small point source which is too close to our actual desired star,
replaces those pixels with the image sigma-clipped median, and then saves
this copy to the folder 'excision' for use in PSF modelling.
"""

import math
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
from photutils.segmentation import detect_threshold, detect_sources, \
                                   deblend_sources, SourceCatalog
from photutils.utils import circular_footprint
from photutils.aperture import SkyCircularAnnulus, aperture_photometry


'HST_resampled/PSF/excision'

def excise(inputdir='HST_resampled/PSF', outputdir='excision', 
           outputdir2='HST_resampled/PSF/excision', copies=['ie4709_drz.fits'],
           cuts=['ie4710_drz.fits'], cutregs=[[[500, 527, 716, 742]]]):
# =============================================================================
# This function has default behavior to suit my needs, but might be 
# generalizeable to copy from a given directory (string inputdir) to another
# given directory (string outputdir, as realtive to inputdir) exact copies of
# a series of fits files (list of strings copies), and altered copies of
# another series of fits files (list of strings cuts).  The files from cuts
# will have a the values in a series of rectangular regions replaced with the
# sigma-clipped median value of their data, respectively (list of lists of
# lists of ints, cutregs).  cutregs is read as: a list wherein each element
# corresponds to a file in cuts, wherein each element corresponds to a region,
# wherein the four elements are integer pixel positions: x_left, x_right, 
# y_bottom, y_top. Also because my programming can be bad sometimes, there also
# needs to be a directory relative to the current directory where the files are
# going (outputdir2)
# =============================================================================
   
   # Exact copies
   for filename in copies:
      os.system("cd %s ; cp %s %s/%s" % (inputdir, filename, outputdir, 
                                         filename))
   
   # Cut files
   for n, filename in enumerate(cuts):
      #os.system("cd %s ; cp %s %s/%s" % (inputdir, filename, outputdir, 
      #                                   filename))
      with fits.open('%s/%s' % (inputdir, filename)) as hdul:
         #data = hdul[1].data
         mean_val, median_val, std_val = sigma_clipped_stats(hdul[1].data, 
                                                             sigma=2.0)
         for reg in cutregs[n]:
            hdul[1].data[reg[2]:reg[3], reg[0]:reg[1]] = median_val
         hdul.writeto('%s/%s' % (outputdir2, filename), overwrite=True)
         #hdul_new = fits.PrimaryHDU(data=data, header=hdul.header)
      #hdul_new.writeto('%s/%s' % (outputdir, filename), overwrite=True)
   return

excise()
