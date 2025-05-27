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

def pull(filename, ex):
   image = get_pkg_data_filename(filename)
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
   plt.savefig('tempfig.png', dpi=300)
   plt.clf()
   print('Figure saved to tempfig.png locally.')

def makeHist(dat, left, right):
   plt.hist(dat.flatten(), bins=300, range=(left, right))
   plt.yscale('log')
   plt.savefig('temphist.png', dpi=300)
   plt.clf()
   print('Histogram saved to temphist.png locally.')

# Input
file = input('File location: ')
extension = int(input('Extension: '))
ifHist = input('Do you need to see a histogram? (Y/N) ')

# Get Data
data = pull(file, extension)

# Histogram
if ifHist == 'Y':
   flat = np.sort(data.flatten())
   cut = 1 / 0.01
   left = flat[int(len(flat) / (0.1 * cut))]
   right = flat[int(-1 * len(flat) / cut)]
   makeHist(data, left, right)
   satCheck = input('Do you want a new histogram with different limits? (Y/N) ')
   if satCheck == 'Y':
      satisfied = False
   else:
      satisfied = True
   while not satisfied:
      left = float(input('Input new lower limit: '))
      right = float(input('Input new upper limit: '))
      makeHist(data, left, right)
      satCheck = input('Do you want a new histogram with different limits? (Y/N) ')
      if satCheck != 'Y':
         satisfied = True

# Plotting
makeSin = input('Use arcsinh scaling? (Y/N) ')
if makeSin == 'Y':
   mSin = True
else:
   mSin = False
limCheck = input('Use custom limits? (Y/N) ')
if limCheck == 'Y':
   left = float(input('Input lower limit: '))
   right = float(input('Input upper limit: '))
else:
   flat = np.sort(data.flatten())
   cut = 1 / 0.01
   left = flat[int(len(flat) / (0.1 * cut))]
   right = flat[int(-1 * len(flat) / cut)]
makeFig(data, left, right, mSin)
