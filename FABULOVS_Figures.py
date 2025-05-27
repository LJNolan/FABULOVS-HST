#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:19:41 2024

@author: ljnolan

Figure Generation for FABULOVS
"""
import math
import os
import numpy as np
from supernolan import plotutils
from affogato import dataPull, getBkgrd, cut, get_guesses, \
                       make_galfit_input, radial_profile, percentile_cut, \
                       errfile, input_to_guess
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import LogStretch, PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
plt.rcParams.update({'font.size': 18})


def complex_task(working_dir, backupdir, outputdir, file, name, coord, size, 
                 scale, frame='icrs', imname='galim.png', center_psf=False,
                 **kwargs):
# =============================================================================
# TODO: Documentation
# write a literal documentation
#
# Note that both working_dir and file need to be as relative to the current
# directory - that which this file is in. Also, size and scale can be of 
# lengths 1 or 2 - length 1 implies square dimensions.
# 
# This function is another attempt at writing a multipurpose function - this
# time an example of using quick-running functions and backup data from prior
# GALFIT runs to create quick images for use in testing or combinations for
# the paper, without the slowdown of running GALFIT.  It follows the general 
# structure of "automate" up until calling a modified "disp_galfit" which then
# has modified dependencies.
# =============================================================================
   path = working_dir
   if len(size)>1:
      size = tuple(size)
   else:
      size = tuple(size[0], size[0])
   if len(scale)==1:
      scale = [scale[0], scale[0]]
   cut(file, coord, size, outputdir=path)
   errfile(file, path)
   cut('%s/errmap.fits' % path, coord, size, outputdir=path,
       outputname='errmap_cut.fits')
   errmap = dataPull('%s/errmap_cut.fits' % path)
   size, zp, sky, comps = get_guesses(path, double=True, masking=True)
   make_galfit_input(size, zp, scale, sky, comps, outputdir=path) # for mask
   loc = (-1, -1)
   if center_psf:
      comps = input_to_guess('%s/%s.galfit.01' % (backupdir, name))
      psf = next(comp for comp in comps if len(comp) == 3)
      loc = (psf[0], psf[1])
   # modified disp_galfit
   disp_galfit(path, backupdir, outputdir, name, save=True, imname=imname,
               scale=scale[0], errmap=errmap, loc=loc, **kwargs)
   return


def disp_galfit(inputdir, backupdir, outputdir, name, save=True, 
                imname='galim.png', thorough=False, masking=False, 
                psfsub=False, radprof=False, scale=0.0, errmap=None, **kwargs):
# =============================================================================
# OLD DESCRIPTION
# This function takes a series of optional arguments: a directory where to find
# the desired galfit run (str inputdir), a boolean which dictates whether to 
# save (save=True) or display the output (False), a directory where to save 
# said output (str outputdir, relative to current directory), a name to save 
# said output under, a boolean dictating whether to just plot the input,
# model, and residual (thorough=False) or plot all the contents of 
# subcomps.fits, and a boolean whether to use a mask (masking). If it's desired
# to add a 5" scalebar, one must supply a pixel scale in arcsec (float scale).
# If using radprof, must supply a coordinate (SkyCoord coord), and kwargs are
# passed to rp_plot Assumes that the GALFIT outputs that will be processed were
# produced following the conventions in this wrapper - i.e. 'galfit.fits' and
# 'subcomps.fits'
# 
# Dependencies: dataPull
#
# NEW DESCRIPTION
# I'm testing using GridSpec to make this work, and making a version where I
# can run test images using existing backup files.
# =============================================================================
   if masking:
      mask = np.ma.make_mask(dataPull('%s/mask.fits' % inputdir))
   else:
      mask = None
   compnum = len(fits.open('%s/%s.subcomps.fits' % (backupdir, name)))
   compnum = range(compnum)[2:]
   cols = 3
   if radprof:
      cols += 1
   if psfsub:
      cols += 1
   arrange = 1+math.ceil(len(compnum)/cols)
   if thorough:
      fig = plt.figure(figsize=(6*cols, 6 * arrange))
   else:
      fig = plt.figure(figsize=(6*cols, 6))
   hr, p = [4,1], arrange-1
   while p>0:
      hr.append(5)
      p-=1
   sp = 0.05 # spacing factor : hspace, wspace
   if thorough:
      gs = GridSpec(arrange+1, cols, wspace=sp, hspace=sp, height_ratios=hr)
   else:
      gs = GridSpec(2, cols, wspace=sp, hspace=sp, height_ratios=hr[0:2])
   # Main components
   n = 1
   while n <= cols:
      # Data processing and cleaning; if n == 3 & psfsub, data is passed down
      # to n == 4
      if n < 4:
         data = dataPull('%s/%s.galfit.fits' % (backupdir, name), n)
         llim, ulim = np.percentile(data, [1, 99])
         if n == 1:
            bkgrd = getBkgrd('%s/%s.galfit.fits' % (backupdir, name), n)
            norm = ImageNormalize(stretch=LogStretch(), vmin=bkgrd, vmax=ulim)
            if psfsub:
               holdover = data - dataPull('%s/%s.subcomps.fits' % \
                          (backupdir, name), compnum[0])
         if n == 3:
            if errmap is None:
               noise = np.std(data[(data>llim) & \
                                (data<ulim)])
            else:
               noise = errmap
         if masking and n!=1:
            if n == 3:
               data = np.where(mask==0, data, 0)
            data = np.ma.masked_array(data, mask=mask)
         datat = percentile_cut(data, 1, 99)
      
      ax = fig.add_subplot(gs[0:2, n-1])
      ax.set_axis_off()
      ax.grid()
      if scale > 0:
         corn = len(data) / 20
         pts = [[corn, corn+(5/scale)], [corn, corn]]
         ax.plot(pts[0], pts[1], 'w-', linewidth=4)
         ax.text(corn+(2.0/scale), corn * (6.5/5), '5"', color='w', 
                 fontweight='bold', fontsize='large')
      origin = 'lower'
      cmap = 'viridis'
      interpolation = 'nearest'
      if n == 1:
         ax.set_title('Data')
         ax.imshow(datat, norm=norm, origin=origin, cmap=cmap, 
                   interpolation=interpolation)
      elif n == 2:
         ax.set_title('Model')
         ax.imshow(datat, norm=norm, origin=origin, cmap=cmap, 
                   interpolation=interpolation)
      elif psfsub and n == 3:
         holdover = np.ma.masked_array(holdover, mask=mask)
         holdovert = percentile_cut(holdover, 1, 99)
         ax.set_title('Data - PSF')
         ax.imshow(holdovert, norm=norm, origin=origin, cmap=cmap, 
                   interpolation=interpolation)
      else:
         ax.set_title('Residual')
         im = ax.imshow(data/noise, origin=origin, vmin=-3, vmax=5,
                        cmap=cmap, interpolation=interpolation)
         divider = make_axes_locatable(ax)
         cax = divider.append_axes('bottom', size='4%', pad=0.02)
         cbar = fig.colorbar(im, cax=cax,orientation='horizontal')
         cbar.ax.tick_params(labelsize=14, width=1.5)
         cbar.set_label(label = r"$\sigma$", fontsize=14, labelpad=-4)
         break
      n += 1
            
   # Radial Profile
   if radprof:
      locators = [0, 1, cols-1]
      rp_plot(fig, gs, locators, inputdir, backupdir, outputdir, name, 
              save=False, **kwargs)
   # Subcomps
   if thorough:
      for n in compnum:
         data = dataPull('%s/%s.subcomps.fits' % (backupdir, name), n)
         if masking:
            data = np.ma.masked_array(data, mask=mask)
         transform = PercentileInterval(99.)
         datat = transform(data)
         norm = ImageNormalize(stretch=LogStretch())
         ax = fig.add_subplot(gs[((n-1)//cols)+2, ((n-1)%cols)-1])
         ax.set_axis_off()
         ax.grid()
         ax.imshow(datat, norm=norm, origin='lower', cmap='Greys_r', 
                   interpolation='nearest')
   
   #if radprof:
      #plt.tight_layout()
   #else:
      #plt.subplots_adjust(wspace=0, hspace=0.05)
   if save:
      plt.savefig('%s/%s.%s' % (outputdir, name, imname), bbox_inches='tight')
   plt.show()
   plt.close()
   return


def rp_plot(fig, gs, locators, inputdir, backupdir, outputdir, name, 
            comps=['unknown'], save=False, imname='radprof.png', flx=False, 
            res=True, ylim=30., **kwargs):
# =============================================================================
# OLD DESCRIPTION
# This function makes a nice radial profile plot on the given axis 
# (matplotlib Axes ax) summarizing the most recent GALFIT run in the optional 
# given directory string (str inputdir), and optionally saves it (toggled by 
# boolean save) to a second optional directory (str outputdir, relative to 
# current directory) under an optional name (str name). The function also takes
# a list of fit component names for the plot (comps) - if none are given,
# assumes a default list. The function by default produces a residual plot
# below the main plot, but this can be toggled by a boolean (res). Plotting is
# affected by a lower limit (float ylim) to prevent absurdly dim outer edges
# from modelling. Also calls for (flx) as per rad_prof_from_file. kwargs are
# passed to rad_prof_from_file.
#
# Dependencies: dataPull, radial_profile, getBkgrd, rad_prof_from_file, 
#               plotutils, add_residual
# =============================================================================
   if comps[0] == 'unknown':
      comps = ['data', 'total model', 'PSF', 'sersic', 'contaminant']
   colors = ['k', '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628',
             '#984ea3','#999999', '#e41a1c', '#dede00']
   compnum = len(fits.open('%s/%s.subcomps.fits' % (backupdir, name)))
   compnum = range(compnum)[2:]
   muss, rad = rad_prof_from_file(inputdir, backupdir, name, flx=flx, 
                                  bksb=True, getr=True, **kwargs)
   x = np.linspace(0, 1, len(muss)) * rad
   muss = [muss]
   muss.append(rad_prof_from_file(inputdir, backupdir, name, file='galfit', 
                                  ext=2, flx=flx, **kwargs))
   # Subcomps
   for n in compnum:
      muss.append(rad_prof_from_file(inputdir, backupdir, name, 
                                     file='subcomps', ext=n, flx=flx,
                                     **kwargs))
   muss = np.asarray(muss)
   
   # Add axis
   if res:
      ax1 = fig.add_subplot(gs[locators[0],locators[2]])
      ax2 = fig.add_subplot(gs[locators[1],locators[2]], sharex=ax1)
   else:
      ax1 = fig.add_subplot(gs[locators[0]:locators[1]+1,locators[2]])
   
   for n, mus in enumerate(muss):
      style = 'solid'
      if n < len(comps):
         color = colors[n]
         label = comps[n]
      else:
         if n < len(colors):
            color = colors[n]
         else:
            color = colors[n-len(colors)]
            style = 'dashed'
            if n > 20:
               print("You modelled with more than 16 contaminants, which " +
                     "this plotting set can't handle.  How did you even do" +
                     "that???")
         label = comps[-1]
      ax1.plot(x, mus, color=color, label=label, linestyle=style)
   
   xlab = "Radial Distance [arcsec]"
   ax1.legend() # Must come before residuals
   
   # Residuals
   if res:
      residual = muss[0] - muss[1]
      # Replacing add_residual function
      flat = np.zeros_like(x) 
      ax2.plot(x, flat, color='k', linewidth=1)
      ax2.scatter(x, residual, color="crimson", marker='x')
      ax2.ticklabel_format(axis='y', style='sci')
      ax2.set_xlabel(xlab)
      ax2.set_ylabel(r'$\Delta\mu$')
      ax2.yaxis.set_label_position("right")
      ax2.yaxis.tick_right()
      ax1.tick_params(axis='x', labelbottom=False)
   else:
      ax1.set_xlabel(xlab)
   ax1.set_xlim(left=0, right=x[-1])
   
   if flx:
      plotutils(ax1, yscale='log')
      ax1.set_ylabel("Flux")
      ax1.set_ylim(bottom=100)
   else:
      plotutils(ax1)
      ax1.set_ylabel(r"$\mu$ [mag arcsec$^{-2}$]")
      ax1.invert_yaxis()
      ax1.set_ylim(bottom=30) # Toggle
   ax1.yaxis.set_label_position("right")
   ax1.yaxis.tick_right()
   if save:
      plt.savefig('%s/%s.%s' % (outputdir, name, imname))
   return


def rad_prof_from_file(inputdir, backupdir, name, file='image', ext=0,
                       loc=(-1,-1), radfrac=1.0, bksb=True, getr=False,
                       **kwargs):
# =============================================================================
# OLD DESCRIPTION
# This function acts as a utility wrapper for radial_profile, pulling the 
# header information from 'image.fits' at the optional directory string
# (inputdir), and data from the optional file string (file) + '.fits' at the
# optional extension int (ext), and then makes a radial profile around a pixel
# point optionally given by a tuple of ints (loc). It assumes the desired
# radius is some float fraction of half the image size (radfrac), and does
# background subtraction (toggled by bksb). Can optionally return radius in 
# arcsec (toggled by getr). kwargs are passed to radial_profile.
#
# Background subtraction is unnecessary when using GALFIT subcomps.
#
# Dependencies: dataPull, radial_profile, getBkgrd
# =============================================================================
   if file == 'image':
      datafile = '%s/image.fits' % inputdir
   else:
      datafile = '%s/%s.%s.fits' % (backupdir, name, file)
   image = '%s/image.fits' % inputdir
   data = dataPull(datafile, ext=ext)
   if bksb:
      bkgrd = getBkgrd(datafile, ext=ext)
      data = data - bkgrd
   hdu = fits.open(image)
   hdr = hdu[0].header
   shp = tuple(hdu[0].data.shape)
   radius = shp[0] * radfrac / 2
   wcs = WCS(hdr)
   if loc == (-1, -1):
      loc = (shp[0]/2, shp[1]/2)
   lat, long = wcs.all_pix2world(loc[0], loc[1], 0)
   ra, dec = lat*u.deg, long*u.deg
   coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
   fscale = hdr['fscale']
   radii = np.linspace(1, radius, num=50)
   zp = -2.5*np.log10(hdr["PHOTFLAM"])-21.10
   exptime = hdr['EXPTIME']
   mus = radial_profile(data, radii, wcs, coord, fscale, zp, exptime=exptime,
                        **kwargs)
   if getr:
      return mus, radius * fscale
   else:
      return mus


def image_auto(c, side, control, psfkey, thorough, psfsub, radprof, outputdir):
   path = 'galfit_automate'
   if not control:
      catalog = Table.read('HST_resampled/catalog.csv', format='csv')
      name = 'ie47%.2i' % c
      file = 'HST_resampled/%s_drz.fits' % name
      frame = 'icrs'
      c-=1
      pos = SkyCoord(ra=catalog['ra'][c]*u.degree, 
                     dec=catalog['dec'][c]*u.degree, frame=frame)
      backupdir = 'backup/soi'
   else:
      catalog = Table.read('Control_Sample/controlsample.csv', 
                           format='csv')
      name = 'control%.3i' % c
      file = 'Control_Sample/sample/%s_drz.fits' % name
      #wherestr = catalog[catalog['myID'] == c]
      #wherestr = wherestr['ra'][0] + ' ' + wherestr['dec'][0]
      #pos = SkyCoord(wherestr, unit=(u.hourangle, u.deg), frame='icrs')
      frame = 'icrs'
      wherecat = catalog[catalog['myID'] == c]
      pos = SkyCoord(ra=wherecat['ra']*u.degree, 
                     dec=wherecat['dec']*u.degree, frame=frame)
      backupdir = 'backup/control'
   size = [side, side]
   scale = [0.06, 0.06]
         
   radfrac = 0.3
   
# =============================================================================
#    if not control:
#       backupdir = 'backup/soi_dedPSF'
#    else:
#       if psfkey == 0:
#          backupdir = 'backup/control_soiPSF'
#       else:
#          backupdir = 'backup/control_selfPSF'
# =============================================================================
   
   complex_task(path, backupdir, outputdir, file, name, pos, size, scale,
                center_psf=True, thorough=thorough, masking=True,
                psfsub=psfsub, radprof=radprof, flx=False, radfrac=radfrac)
   return

working_dir = 'misc/image_tests'
c = 1 # which file?
startingk = 0 # midstart?

# Notes:
# Sample of Interest: 1-8
# 4 and 6 are bigger (at min 400x400)
# 3 is badly behaved and won't converge because of an overlapping guy,
# 5 & 7 also doesn't want to converge for no clear reason.

side = 400
control = True
compare = False
thorough = False
psfsub = True
radprof = True
loop = True
check = False
skipbad = True
if control:
   sample = [2, 3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 44, 47, 48, 70, 80, 84, 
             85, 86, 87]
   samplekeys = [0, 4, 2,  1,  4,  1,  0,  0,  2,  3,  3,  0,  3,  0,  4,  2,
                 2,  4,  2,  1]
   badagn = [5, 16, 19, 22, 33, 37, 47, 70, 80, 84, 86, 87]
else:
   sample = [i+1 for i in range(8)]
   psfkey = 0
if loop:
   for k, b in enumerate(sample):
      if k < startingk:
         continue
      sd = side
      if not control:
         if b in (4, 6):
            sd += 200
         psfkey = 0
      else:
         if skipbad and (b in badagn):
            continue
         psfkey = 0#samplekeys[k] # For now just running on SoI PSF
      image_auto(b, sd, control, psfkey, thorough, psfsub, radprof,
                 working_dir)
      if control and compare and psfkey != 0:
         image_auto(b, sd, control, 0, thorough, radprof, working_dir)
      if check and b != sample[-1]:
         ip = 'f'
         while ip not in ['y', 'n', 'Y', 'N']:
            print("Continue? [y/n]:")
            ip = input()
         if ip in ['n', 'N']:
            print('Your startingk value is ', k+1)
            break
else:
   image_auto(c, side, control, psfkey, thorough, psfsub, radprof,
              working_dir)
