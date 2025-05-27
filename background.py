#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:29:27 2023

@author: ljnolan
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch, PercentileInterval, astropy_mpl_style
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources
from photutils.utils import circular_footprint
plt.style.use(astropy_mpl_style)


def pull(filename, ex):
   image = get_pkg_data_filename(filename)
   image_data = fits.getdata(image, ext=ex)
   return image_data

def showHeader(filename, arg='bees'):
# =============================================================================
# Old function, not yet implemented
# =============================================================================
   if arg=='bees':
      hdu = fits.open(filename + '.fits')[0]
      print(hdu.header)
   else:
      hdu = fits.open(filename + '.fits')[0]
      print(hdu.header[arg])
   return


data = pull('images/4701cut.fits', 0)

transform = PercentileInterval(99.)
datat = transform(data)
norm = ImageNormalize(stretch=LogStretch())

fig = plt.figure(figsize=(18, 12))

ax = fig.add_subplot(2, 3, 1)
ax.imshow(datat, norm=norm, origin='lower', cmap='Greys_r', interpolation='nearest')

sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
threshold = detect_threshold(data, nsigma=3.0, sigma_clip=sigma_clip)
segment_img = detect_sources(data, threshold, npixels=50)
segment_deb = deblend_sources(data, segment_img, npixels=50, 
                               nlevels=32, contrast=0.01, progress_bar=False)

ax = fig.add_subplot(2, 3, 2)
ax.imshow(segment_deb, origin='lower', cmap=segment_img.cmap, interpolation='nearest')

footprint = circular_footprint(radius=10)
mask = segment_img.make_source_mask(footprint=footprint)

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.ma.masked_array(data, mask=mask), origin='lower', cmap='Greys_r', interpolation='nearest')

mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
print((mean, median, std))

data_sub = data - mean

threshold = detect_threshold(data_sub, nsigma=3.0, sigma_clip=sigma_clip)
segment_img = detect_sources(data_sub, threshold, npixels=50)
segment_deb = deblend_sources(data_sub, segment_img, npixels=50, 
                               nlevels=32, contrast=0.005, progress_bar=False)

ax = fig.add_subplot(2, 3, 4)
ax.imshow(segment_deb, origin='lower', cmap=segment_img.cmap, interpolation='nearest')

plt.subplots_adjust(wspace=0, hspace=0.05)

#showHeader('HST_resampled/ie4701_drz', arg='photzpt')
