#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:35:36 2024
Author: Liam Nolan

Based in large part on a code by Tony Chen, as well as photutils documentation.

General Notes for Use:
The general strategy is running prep_verify blindly, and then using the image 
output to make list_index_stars_good for a run of genPSF which actually
makes the PSF.
"""

import os
import math
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.visualization import simple_norm
from astropy.wcs import WCS,utils
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from photutils.detection import find_peaks
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder


def plot_peak_detection(data, peaks_tbl, name, todir='.'):
# =============================================================================
# ADD DOCUMENTATION
# This function 
# =============================================================================
   norm = simple_norm(data, 'sqrt', percent=99.0)
   plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
   plt.grid(False)
   plt.scatter(peaks_tbl["x_peak"], peaks_tbl["y_peak"], c='r', marker='x')
   for i in range(len(peaks_tbl["x_peak"])):
      plt.text(peaks_tbl["x_peak"][i], peaks_tbl["y_peak"][i], str(i), c='r', 
               va='bottom', ha='center')
   plt.savefig("%s/%s_peak_detections.jpeg" % (todir, name), dpi=200)
   plt.close()
   return


def make_stars_tbl(data, peaks_tbl, imgsize):
# =============================================================================
# ADD DOCUMENTATION
# This function 
# =============================================================================
   hsize = (imgsize - 1) / 2
   x = peaks_tbl['x_peak']
   y = peaks_tbl['y_peak']
   mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
           (y > hsize) & (y < (data.shape[0] -1 - hsize)))

   stars_tbl = Table()
   stars_tbl['x'] = x[mask]
   stars_tbl['y'] = y[mask]

   return stars_tbl


def plot_star_cutout(stars, index_stars_good, name, todir='.'):
# =============================================================================
# ADD DOCUMENTATION
# This function 
# =============================================================================
   nrows,ncols = int(math.ceil(len(stars)**0.5)),\
                 int(math.ceil(len(stars)**0.5))
   fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))
   if len(stars) == 1:
      ax = [ax]
   else:
      ax = ax.ravel()
   for i in range(len(stars)):
      norm = simple_norm(stars[i], 'log', percent=99.0)
      ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
      ax[i].text(0.05, 0.95, str(i), fontsize=16, transform=ax[i].transAxes, 
                 c='r', horizontalalignment='left', verticalalignment='top')
      ax[i].tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
      ax[i].grid(False)
      if i in index_stars_good:
         ax[i].text(0.5, 0.95, "V", fontsize=20, transform=ax[i].transAxes, 
                    c='r', horizontalalignment='center', 
                    verticalalignment='top')
   for i in range(len(stars), nrows*ncols):
      ax[i].set_visible(False)

   plt.tight_layout()
   plt.savefig("%s/%s_star_cutouts.jpeg" % (todir, name), dpi=200)
   plt.close()
   return


def save_psf_fits(data, todir='.', psf_name='psf_combined'):
# =============================================================================
# This function takes a data output from an EPSFModel object, i.e. epsf.data,
# which I believe is a 2D array (data), an optional directory string where to
# save the output file (todir), and the name string to prepend to the output
# file (psf_name).  It saves the given data as a fits file with no meaningful
# header information, meant to be read as a PSF model by GALFIT or other
# programs.
# =============================================================================
   outputname = '%s/%s.fits' % (todir,psf_name)
   if os.path.exists(outputname): os.system('rm %s' % outputname)
   hdu = fits.PrimaryHDU(data)
   hdu.writeto(outputname)
   return


def identify_stars(files, list_index_stars_good, imgsize, todir='.', 
                   genpeaks=True, plot=True):
# =============================================================================
# ADD DOCUMENTATION
# This function 
# =============================================================================
   names = [file.split('/')[-1].rsplit('.',1)[0] for file in files]
           # used for labelling of subproducts
   list_nddata,list_good_stars_tbl = [],[]
   for i in range(len(files)):
      file = files[i]
      name = names[i]
      index_stars_good = list_index_stars_good[i]
      hdu = fits.open(file)
      wcs = WCS(hdu[1].header)
      data = hdu[1].data

      # calculate rms
      boundary_rms = np.nanpercentile(data, 0), np.nanpercentile(data, 95)
      rms = np.std(data[(data>boundary_rms[0]) & (data<boundary_rms[1]) ])
   
      # find or import peaks
      if genpeaks: 
         peaks_tbl = find_peaks(data, threshold=1000*rms, box_size=20, 
                                npeaks=16, border_width=imgsize//2, wcs=wcs)
         # if throwing weird errors, remove wcs=wcs above
         peaks_tbl.write('%s/%s_peaks_tbl.dat' % (todir, name), overwrite=True,
                         format='ascii')
      else:
         peaks_tbl = Table.read('%s/%s_peaks_tbl.dat' % (todir, name),
                                format='ascii')
      if plot:
         plot_peak_detection(data, peaks_tbl, name, todir=todir)
      
      # make the list of star cutout   
      stars_tbl = make_stars_tbl(data, peaks_tbl, imgsize)
   
      # remove the background
      mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)  
      data -= median_val  
   
      # make nddata for extract_stars func
      nddata = NDData(data=data)
   
      # extract stars' cutouts and mark good stars
      stars = extract_stars(nddata, stars_tbl, size=imgsize)  
      if plot:
         plot_star_cutout(stars, index_stars_good, name, todir=todir)
      
      # append nddata and proper portions of stars
      list_nddata.append(nddata)
      list_good_stars_tbl.append(stars_tbl[index_stars_good])
      
      # create the big return
      all_good_stars = extract_stars(list_nddata, list_good_stars_tbl, 
                                     size=imgsize)

   return all_good_stars


def prep_verify(files, imgsize, todir):
# =============================================================================
# ADD DOCUMENTATION
# This function takes in a list of string filenames of paths to FITS images
# (files), the size of the desired final PSF, in pixels (imgsize), and the
# directory to save the candidate PSFs to as a string (todir). The function
# will identify all the candidate PSF stars in the images passed to it, which
# should then be visually inspected by the user. The order of these peaks is
# saved in the same output directory to be used by genPSF. This also means
# prep_verify should not be re-run between generation of list_index_stars_good
# for genPSF from prep_verify and its use for genPSF, as stochastic behavior
# means the orders seem to fluctuate.
# =============================================================================
   list_index_stars_good = []
   for i in range(len(files)):
      list_index_stars_good.append([])
   identify_stars(files, list_index_stars_good, imgsize, todir=todir,
                  genpeaks=True, plot=True)
   return


def genPSF(files, list_index_stars_good, imgsize, todir,
           psf_name='psf_combined'):
# =============================================================================
# This function takes in a list of string filenames of paths to FITS images
# (files), a same-ordered list of lists of integer indices of the good PSF 
# stars to be pulled from those images (list_index_stars_good), the size of the
# desired final PSF, in pixels (imgsize), the directory to save the final PSF 
# and other outputs to as a string (todir), and a name to prepend to the 
# outputs as a string (psf_name). The function uses a specified number of 
# interations default 10, see maxiters below) to build the effective PSF out of
# the good PSF stars identified using prep_verify and passed using 
# list_index_stars_good. For the user's convenience, both a FITS file of the 
# ePSF and a .jpeg is saved for viewing. It should be noted that this function 
# assumes prep_verify has literally been run - it produces a data file storing
# the order of peaks that lines up the list_index... order properly between
# runs.
# =============================================================================
   all_good_stars = identify_stars(files, list_index_stars_good, imgsize, 
                                   todir=todir, genpeaks=False, plot=False)

   epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                               progress_bar=False)
   epsf, fitted_stars = epsf_builder(all_good_stars) 

   norm = simple_norm(epsf.data, 'log', percent=99.0)
   plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
   plt.colorbar()
   plt.savefig("%s/%s.jpeg" % (todir,psf_name),dpi=200)
   plt.clf()

   # Save the PSF star fits
   save_psf_fits(epsf.data, todir=todir, psf_name=psf_name)
   return
