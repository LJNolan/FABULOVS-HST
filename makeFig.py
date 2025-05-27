#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:00:06 2022

@author: ljnolan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

def pull(num, ex):
   image = get_pkg_data_filename('images/' + num + 'gal.fits')
   image_data = fits.getdata(image, ext=ex)
   return image_data

def makeFig(dat, left, right, isSin=True):
   plt.figure()
   if isSin:
      dat = np.arcsinh(dat)
      plt.imshow(dat, cmap='viridis', vmin=np.arcsinh(left), vmax=np.arcsinh(right))
   else:
      plt.imshow(dat, cmap='viridis', vmin=left, vmax=right)
   #plt.colorbar()
   plt.gca().invert_yaxis()
   plt.axis('off')
   plt.show()
   plt.clf()

def makeHist(dat, left, right):
   plt.hist(dat.flatten(), bins=300, range=(left, right))
   plt.yscale('log')
   plt.show()
   plt.clf()

# Input
file = input('File location: ')
extension = input('Extension: ')
ifHist = input('Do you need to see a histogram? (Y/N) ')

# Get Data
data = pull('4701', 2)

# Which Plot?
makeFig(data, 400, 2500)

#makeHist(data, -500, 1000)
