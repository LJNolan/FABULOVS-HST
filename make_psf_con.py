"""
Created on Jan 22 2024
Author: Liam Nolan

Based in large part by a code by Tony Chen, duplicated from my make_psf_soi

General Notes for Use:
The general strategy is running make_good_star_cutouts blindly, and then using
the image output to make index_stars_good for a second run which actually
makes the PSF.

Note: I may have deleted the HST files this code references when cleaning up.
"""

import os
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

   norm = simple_norm(data, 'sqrt', percent=99.0)
   plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
   plt.scatter(peaks_tbl["x_peak"], peaks_tbl["y_peak"], c='r', marker='x')
   for i in range(len(peaks_tbl["x_peak"])):
      plt.text(peaks_tbl["x_peak"][i], peaks_tbl["y_peak"][i], str(i), c='r', 
               va='bottom', ha='center')
   plt.savefig("%s/%s_peak_detections.jpeg" % (todir, name), dpi=200)
   plt.close()
   return


def make_stars_tbl(data, peaks_tbl, imgsize):

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

   nrows,ncols = int(len(stars)**0.5)+1,int(len(stars)**0.5)+1
   fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))
   ax = ax.ravel()
   for i in range(len(stars)):
      norm = simple_norm(stars[i], 'log', percent=99.0)
      ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
      ax[i].text(0.05, 0.95, str(i), fontsize=16, transform=ax[i].transAxes, 
                 c='r', horizontalalignment='left', verticalalignment='top')
      ax[i].tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
      if i in index_stars_good:
         ax[i].text(0.5, 0.95, "V", fontsize=20, transform=ax[i].transAxes, 
                    c='r', horizontalalignment='center', 
                    verticalalignment='top')

   plt.tight_layout()
   plt.savefig("%s/%s_star_cutouts.jpeg" % (todir, name), dpi=200)
   plt.close()
   return


def save_psf_fits(data, todir='.'):

   outputname = '%s/psf_combined.fits' % todir
   if os.path.exists(outputname): os.system('rm %s' % outputname)
   hdu = fits.PrimaryHDU(data)
   hdu.writeto(outputname)
   return


def make_good_star_cutouts(name, index_stars_good, imgsize, fromdir='.', 
                           todir='.'):

   filename = '%s/%s_drz.fits' % (fromdir, name)
   hdu = fits.open(filename)
   wcs = WCS(hdu[1].header)
   data = hdu[1].data

   # calculate rms

   boundary_rms = np.nanpercentile(data, 0), np.nanpercentile(data, 95)
   rms = np.std(data[(data>boundary_rms[0]) & (data<boundary_rms[1]) ])
   print("rms = %.1f" % rms)

   # find peaks
    
   peaks_tbl = find_peaks(data, threshold=1000*rms, box_size=20, npeaks=16,
                          border_width=imgsize//2)   # 25->imgsize/2?
   plot_peak_detection(data, peaks_tbl, name, todir=todir)

   # make the list of star cutout   

   stars_tbl = make_stars_tbl(data, peaks_tbl, imgsize)

   # remove the background
   mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)  
   data -= median_val  

   # make nddata for extract_stars func
   nddata = NDData(data=data)

   # extract stars' curouts and mark good stars
   stars = extract_stars(nddata, stars_tbl, size=imgsize)  
   plot_star_cutout(stars, index_stars_good, name, todir=todir)
   #good_stars = extract_stars(nddata, stars_tbl[index_stars_good], 
   #size=imgsize)

   #return good_stars
   return nddata, stars_tbl[index_stars_good]


def doit1(c):
   imgsize = 121 # pixel (must be odd)
   fromdir = 'Control_Sample/sample'
   todir = 'misc/PSF_Work'
   name = 'control0%.2i' % c
   index_stars_good = []
   make_good_star_cutouts(name, index_stars_good, imgsize, fromdir=fromdir, 
                          todir=todir)
   return


def doit2():
   imgsize = 161 # pixel
   cons = [3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 47, 48, 70, 80, 84, 85, 86, 
           87]
   fromdir = 'Control_Sample/sample'
   todir = 'misc/PSF_Work'
   names = []
   for c in cons:
      names.append('control0%.2i' % c)
   
   # Checked list of stars which look good from doit1()
   list_index_stars_good = [[1, 2],[0, 3, 4, 5],[0,2,3],[6,7,8,9],
                            [2,3,4,7,8,10],[0],[1],[0,1,2,3,6],[1,7],[0,1,2],
                            [0,2],[0],[3,4,8,9],[2,3,6,12],
                            [0,1,2,4,5,7,8,10,11,12,13,15],
                            [0,2,7,8,9,11,13,14],[0,1,2,3,4,5,6,7,8,9,11,14]]
   
   # select and cut the good PSF stars from each image
   for i in range(len(names)):
      list_nddata,list_good_stars_tbl = [],[]
      nddata, good_stars_tbl = make_good_star_cutouts(names[i],
                                                      list_index_stars_good[i],
                                                      imgsize, fromdir=fromdir,
                                                      todir=todir)
      list_nddata.append(nddata)
      list_good_stars_tbl.append(good_stars_tbl)
      
      # here I'm jury-riggin my PSF code to have it run on a "list" of just one
      # image's best PSF stars, and run on each image.
      
      all_good_stars = extract_stars(list_nddata, list_good_stars_tbl, 
                                     size=imgsize)

      epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                                 progress_bar=False)   # should up maxiters
      epsf, fitted_stars = epsf_builder(all_good_stars) 

      norm = simple_norm(epsf.data, 'log', percent=99.0)
      plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
      plt.colorbar()
      plt.savefig("%s/%s_PSF.jpeg" % (todir,names[i]),dpi=200)
      plt.clf() # maybe remove this if stuff turns up blank

      # Save the PSF star fits in its own folder
      save_psf_fits(epsf.data, todir=todir + '/' + names[i])
   return


# Definitely not valid: 2, 44
# Cringe:               3, 22, 25, 70
# Based?:               5, 14, 16, 19, 33, 37, 40?, 47?, 48, 80, 84, 86, 87
cons = [2, 3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 44, 47, 48, 70, 80, 84, 85, 
        86, 87]
#for n in cons:
#   doit1(n)
doit2()
