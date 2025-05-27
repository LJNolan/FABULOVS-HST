#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:50:04 2024

Author: Liam Nolan

This is the specific implementation of make_psf for FABULOVS
"""
from supernolan import *
from affogato import automate
from make_psf import prep_verify, genPSF
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

imgsize = 161 # pixels, must be odd
consample = [3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 47, 48, 70, 80, 84, 
             85, 86, 87] # 2, 44 both dont have any possible candidates


def doit_pv(c, control):
   if not control:
      files = []
      for b in range(c, 11): # enter c as 1 for full soi, 9 for ded. PSF
         name = 'ie47%.2i' % b
         if b > 8:
            files.append('HST_resampled/PSF/excision/%s_drz.fits' % name)
         else:
            files.append('HST_resampled/%s_drz.fits' % name)
      todir = 'psf_automate/soi'
   else:
      name = 'control%.3i' % c
      files = ['Control_Sample/sample/%s_drz.fits' % name]
      todir = 'psf_automate/control%.3i' % c
   prep_verify(files, imgsize, todir)
   return


def doit_gpsf(c, control, list_index_stars_good, **kwargs):
   if not control:
      files = []
      for b in range(c, 11): # enter c as 0 for full soi, 9 for ded. PSF
         name = 'ie47%.2i' % b
         files.append('HST_resampled/%s_drz.fits' % name)
      todir = 'psf_automate/soi'
   else:
      name = 'control%.3i' % c
      files = ['Control_Sample/sample/%s_drz.fits' % name]
      todir = 'psf_automate/control%.3i' % c
   genPSF(files, list_index_stars_good, imgsize, todir, **kwargs)
   return


def doall_control_pv():
   for c in consample:
      doit_pv(c, True)
   return


def doall_control_gpsf(quality):
   c_ll_isg = control_megalist[4 - quality]
   psf_name = 'q%i_psf_combined' % quality
   for i, c in enumerate(consample):
      if len(c_ll_isg[i][0]) > 0:
         doit_gpsf(c, True, c_ll_isg[i], psf_name=psf_name)
   return


# Quality Winnowing
control_megalist = []
# Including Quality 4
soi_lisg = [[],[0],[3,5],[0,2],[],[1],[1],[1,2,3],[0],[0]]
#soi_lisg = [[0],[0]] # dedicated PSF exposures
control_megalist.append([[[1,2]], #3
            [[0,3,4]], #5
            [[0,2,3]], #14
            [[6,7,8,9]], #16
            [[0,2,3,4,7,8,10,11]], #19
            [[0]], #22
            [[1]], #25
            [[0,1,2,3,6]], #33
            [[1,6,7]], #37
            [[0,1,2]], #40
            [[0,2]], #47
            [[0]], #48
            [[3,4,8,9]], #70
            [[2,3,4,5,11,12]], #80
            [[0,1,2,4,6,7,9,10,11,12,14]], #84
            [[3,4]], #85
            [[0,2,4,7,8,9,11,13,14]], #86
            [[0,1,2,3,4,5,7,11,13]]]) #87
#Including Quality 3
#soi_lisg = [[],[0],[3,5],[0,2],[],[],[1],[2],[0],[0]]
#soi_lisg = [[0],[0]] # dedicated PSF exposures
control_megalist.append([[[]], #3 REMOVED
            [[0,3]], #5
            [[0,2,3]], #14 NO DIFFERENCE
            [[]], #16 REMOVED
            [[2,3,7,8,10]], #19
            [[]], #22 REMOVED
            [[]], #25 REMOVED
            [[0,3,6]], #33
            [[1,6]], #37
            [[1,2]], #40
            [[0,2]], #47 NO DIFFERENCE
            [[0]], #48 NO DIFFERENCE
            [[4]], #70
            [[2,3,5,11,12]], #80
            [[0,1,2,6,7,9,11,14]], #84
            [[3]], #85
            [[0,2,8,11,13,14]], #86
            [[0,1,2,3,7]]]) #87
#Including Quality 2
#soi_lisg = [[],[0],[3,5],[],[],[],[],[],[0],[0]]
#soi_lisg = [[0],[0]] # dedicated PSF exposures
control_megalist.append([[[]], #3
            [[0,3]], #5 NO DIFFERENCE
            [[0,2,3]], #14 NO DIFFERENCE
            [[]], #16
            [[2,3,7,8,10]], #19 NO DIFFERENCE
            [[]], #22
            [[]], #25
            [[0,3,6]], #33 NO DIFFERENCE
            [[]], #37 REMOVED
            [[]], #40 REMOVED
            [[2]], #47
            [[0]], #48 NO DIFFERENCE
            [[]], #70
            [[2,3,5,11,12]], #80 NO DIFFERENCE
            [[0,1,2,6,7,9,11,14]], #84 NO DIFFERENCE
            [[]], #85 REMOVED
            [[2,8,11,13,14]], #86
            [[0,1,2,3,7]]]) #87 NO DIFFERENCE
#Including Quality 1
#soi_lisg = [[],[],[5],[],[],[],[],[],[0],[0]]
#soi_lisg = [[0],[0]] # dedicated PSF exposures
control_megalist.append([[[]], #3
            [[]], #5 REMOVED
            [[2,3]], #14
            [[]], #16
            [[2,3]], #19
            [[]], #22
            [[]], #25
            [[6]], #33
            [[]], #37
            [[]], #40
            [[]], #47 REMOVED
            [[]], #48 REMOVED
            [[]], #70
            [[]], #80 REMOVED
            [[1]], #84
            [[]], #85
            [[8]], #86
            [[0,1,2,3]]]) #87

c = 1 # controls soi: 1 for all, 9 for ded. PSF exp.  can also select con.
quality = 1 # controls control

#doit_pv(c, False)  # Make quality check images for SOI
#doall_control_pv() # Make quality check images for entire control sample
#doit_gpsf(c, False, soi_lisg) # Make PSF for SOI
doall_control_gpsf(quality)  # Make PSF for entire control sample
