#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:12:09 2024

@author: ljnolan

This is the merger code where I bring together my PSF and GALFIT wrapper codes.
"""
from supernolan import *
from affogato import automate
from make_psf import prep_verify, genPSF
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

def run_auto(c, side, control, thorough, radprof, bkp):
   path = 'galfit_automate'
   if not control:
      catalog = Table.read('HST_resampled/catalog.csv', format='csv')
      name = 'ie47%.2i' % c
      file = 'HST_resampled/%s_drz.fits' % name
      frame = 'icrs'
      c-=1
      pos = SkyCoord(ra=catalog['ra'][c]*u.degree, 
                     dec=catalog['dec'][c]*u.degree, frame=frame)
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
   size = [side, side]
   scale = [0.06, 0.06]
   psf = '../misc/PSF_Work/clean_psf_combined.fits'
   radfrac = 0.3
   #imname = 'galim.png' # using defaults
   backupdir = None
   if bkp:
      backupdir = '../misc'
   
   automate(path, file, pos, size, scale, psf, backupdir=backupdir, 
            bkpname=name, thorough=thorough, masking=True, radprof=radprof,
            flx=False, radfrac=radfrac)
   return


c = 2 # which file?

# Notes:
# Sample of Interest: 1-8
# 4 and 6 are bigger (at min 400x400)
# 3 is badly behaved and won't converge because of an overlapping guy,
# 5 & 7 also doesn't want to converge for no clear reason.

side = 400
control = False
thorough = False
radprof = True
loop = False
check = True
if control:
   sample = [2, 3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 44, 47, 48, 70, 80, 84, 
             85, 86, 87]
else:
   sample = [i+1 for i in range(8)]
if loop:
   bkp = True
   for b in sample:
      sd = side
      if not control:
         if b in (4, 6):
            sd += 200
      run_auto(b, sd, control, thorough, radprof, bkp)
      if check and b != sample[-1]:
         ip = 'f'
         while ip not in ['y', 'n', 'Y', 'N']:
            print("Continue? [y/n]:")
            ip = input()
         if ip in ['n', 'N']:
            break
else:
   bkp = False
   run_auto(c, side, control, thorough, radprof, bkp)
