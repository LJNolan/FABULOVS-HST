#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:41:57 2024

@author: ljnolan

This is my pipeline for statmorph to get statistics n such.
"""

#from supernolan import *
from affogato import cut, dataPull, getBkgrd, to_latex
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
from photutils.segmentation import detect_threshold, detect_sources, \
                                   make_2dgaussian_kernel, SourceCatalog, \
                                   deblend_sources
import statmorph as stm
from astropy import units as u

def get_stats(image, coords=None, gain=1, segmap_deb=None, indices=None):
# =============================================================================
# This function takes in a background-subtracted image (numpy array image), an
# optional list of coordinates in pixels (coords), an optional float gain 
# (gain, default 1), an optional deblended segmentation map (segmap_deb), and 
# list of integer labels from said segmentation map to produce statistics 
# (indices). It outputs the asymmetry, shape asymmetry, and Gini-M20 statistic
# of the sources whose centroid is closest to each coordinate, OR those sources
# which are indicated in the given map. Note that one must provide either 
# 'coords' or both 'segmap_deb' and 'indices'.
#
# Dependencies: dataPull, getBkgrd
# =============================================================================
   if coords is None and segmap_deb is None:
      print("Fatal error: must provide either list of coords or segmentation", 
            "map.")
      return
   elif coords is None and indices is None:
      print("Fatal error: must provide list of indices if providing", 
            "segmentation map.")
      return
   elif coords is not None and segmap_deb is not None:
      print("Warning: coords and segmap_deb provided, using coords.")
   if coords is not None:
      segmap_deb, indices = make_seg(image, coords)
   
   source_morphs = stm.source_morphology(image, segmap_deb, gain=gain)
   stats_list = []
   for n in indices:
      morph = source_morphs[n-1]
      stats_list.append([n, morph.asymmetry, morph.shape_asymmetry,
                         morph.gini_m20_merger])
   return stats_list


def make_seg(image, coords, thresh_sig=1.5, npixels=50, kernel_FWHM=3.0, 
             kernel_size=5):
# =============================================================================
# This function takes in a background-subtracted image (numpy array image), and
# a list of coordinates in pixels (coords), and produces a segmentation map and
# the indices of the closest objects to the given coords.
#
# Dependencies: 
# =============================================================================
   threshold = detect_threshold(image, thresh_sig)
   kernel = make_2dgaussian_kernel(kernel_FWHM, size=kernel_size)
   image_conv = convolve(image, kernel)
   segmap = detect_sources(image_conv, threshold, npixels)
   segmap_deb = deblend_sources(image_conv, segmap, npixels=npixels, 
                                 nlevels=32, contrast=0.02, progress_bar=False)
   segmap_clean, indices = clean_seg(image_conv, segmap_deb, coords)
   return segmap_deb, indices


def clean_seg(data, seg_im, coords):
# =============================================================================
# This function takes a data array (data), a photutils segmentation image
# from that data (seg_im), and a desired coordinate or list of coordinates as a
# numpy array (coords) and returns a modified segmentation image removing all 
# sources except those with centroids closest to the given coordinate(s), as 
# well as a list of those sources' labels.
# Note: Modified from FABULOVS_GALFIT implementation of get_mask.
# =============================================================================
   cat = SourceCatalog(data, seg_im)
   tab = cat.to_table(['xcentroid', 'ycentroid', 'label'])
   if coords.shape == (2,):
      coords = np.asarray([coords])
   goods = []
   for coord in coords:
      locs = np.zeros((len(tab), 2))
      locs[:,0], locs[:,1] = tab['xcentroid'], tab['ycentroid']
      tab['dist'] = np.linalg.norm(locs - coord, axis=1)
      tab.sort('dist')
      goods.append(tab[0]['label'])
      tab = tab[1:]
   tab = tab['label']
   cleaned_seg = seg_im.copy()
   for n in range(len(tab)):
      lab = int(tab[n])
      cleaned_seg.data[cleaned_seg.data == lab] = 0
   return cleaned_seg, goods


def get_image(c, size, scale, control, backupdir=None, psfsub=True):  
# =============================================================================
# This function takes an integer label (c), a side length list in pixels
# (side), a list scale factor in arcsec/pixel (scale, length 1 or 2 - 1 implies
# square), a boolean toggle for if this is or is not running on the control
# sample (control), an optional string directory to look for a backup of
# data - which implies the desired result is from a residual produced by
# BigWrapper (backupdir) - and a boolean toggle between using the total model
# residual and only the psf-subtraction (psfsub), and outputs a numpy array of
# the background-subtracted data in a stamp of the given size (image). 'c'
# takes different meanings depending on if 'control' is True or not, it is my
# way of differentiating members of the sample of interest and the control
# sample - each have numbers but those numbers mean different things.  You can
# replace all the code up to the place indicated with something that works for
# your data storage, you just need to define path (where you want temporary 
# files stored), file (a string path to the file being cut from), and pos (a
# SkyCoord position to do the cut).
#
# Another explanation because I'm McLosing my McMind - no backupdir means
# return the true data cutout.  With backupdir, psfsub=True means return true
# data cutout minus PSF model from GALFIT, and psfsub=False means return the
# GALFIT residual (which is already data - full model).  HOWEVER, I discovered
# doing stats on the residual is essentially a fool's errand because Statmorph
# can't handle regions that have net-negative sums, which can happen in a
# residual which one expects to average ~0.
# =============================================================================
   path = 'stats_automate'
   if not control:
      catalog = Table.read('HST_resampled/catalog.csv', format='csv')
      name = 'ie47%.2i' % c
      file = 'HST_resampled/%s_drz.fits' % name
      frame = 'icrs'
      pos = SkyCoord(ra=catalog['ra'][c-1]*u.degree, 
                     dec=catalog['dec'][c-1]*u.degree, frame=frame)
   else:
      catalog = Table.read('Control_Sample/controlsample.csv', 
                           format='csv')
      name = 'control%.3i' % c
      file = 'Control_Sample/sample/%s_drz.fits' % name
      frame = 'icrs'
      wherecat = catalog[catalog['myID'] == c]
      pos = SkyCoord(ra=wherecat['ra']*u.degree, 
                     dec=wherecat['dec']*u.degree, frame=frame)
   # ===== here ends the computer-specific stuff ======
   if len(size)>1:
      size = tuple(size)
   else:
      size = tuple(size[0], size[0])
   if len(scale)==1:
      scale = [scale[0], scale[0]]
   cut(file, pos, size, outputdir=path)
   stamp = '%s/image.fits' % path
   image = dataPull(stamp)
   background = getBkgrd(stamp)
   image_sub = image - background
   if backupdir is None:
      pass
   else:
      if psfsub:
         psf = dataPull('%s/%s.subcomps.fits' % (backupdir, name), ext=2)
         plt.imshow(psf, cmap='gray', origin='lower',
           norm=simple_norm(image, stretch='log', log_a=10000))
         image_sub = image_sub - psf
      else:
         file = '%s/%s.galfit.fits' % (backupdir, name)
         image_sub = dataPull(file, ext=3)
   return image_sub


def automate1(c, side, scale, control, res):
   size = [side, side]
   true_image = get_image(c, size, scale, control)
   center = np.asarray(size) / 2
   if res:
      if not control:
         backupdir = 'backup/soi'
      else:
         backupdir = 'backup/control'
      image2 = get_image(c, size, scale, control, backupdir=backupdir,
                         psfsub=True)
      segmap_deb, indices = make_seg(true_image, center)
      stuff = get_stats(image2, segmap_deb=segmap_deb, indices=indices)[0]
   else:
      stuff = get_stats(true_image, center)[0]
   stuff[0] = c # replacing index with c identifier
   # Adding proper name to stats list
   if not control:
      catalog = Table.read('HST_resampled/catalog.csv', format='csv')
      fullname = catalog['name'][c-1]
   else:
      catalog = Table.read('Control_Sample/controlsample.csv', 
                           format='csv')
      name = 'control%.3i' % c
      fullname = catalog[catalog['myID'] == c]['SDSS Name'][0]
   stuff.insert(0, fullname)
   return stuff


def automate2(control, skipbad=True):
   heads = ['Name', 'ID', 'Net Asym', 'Net SAsym', 'Net GM20', 'Sub Asym',
            'Sub SAsym', 'Sub GM20']
   stats = []
   scale = [0.06]
   if not control:
      tabname = 'stats_soi.tab'
      for c in range(9)[1:]:
         side = 400
         if c in [4,6]:
            side += 200
         stuff = automate1(c, side, scale, control, False)
         stuff += automate1(c, side, scale, control, True)[2:]
         stats.append(stuff) # stuff[0:1]+stuff[2:] to skip ID
   else:
      tabname = 'stats_con.tab'
      sample = [2, 3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 44, 47, 48, 70, 80,
                84, 85, 86, 87]
      badagn = [5, 16, 19, 22, 33, 37, 47, 70, 80, 84, 86, 87]
      for c in sample:
         if skipbad:
            if c in badagn:
               continue
         side = 400
         stuff = automate1(c, side, scale, control, False)
         stuff += automate1(c, side, scale, control, True)[2:]
         stats.append(stuff) # stuff[0:1]+stuff[2:] to skip ID
   tab = Table(names=heads, rows=stats)
   # Consider rounding values
   tab.round(4)
   to_latex(tab, outputdir='backup', tabname=tabname)
   return tab


control = False
tab = automate2(control)
