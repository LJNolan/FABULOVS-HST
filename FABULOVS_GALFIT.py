#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:12:09 2024

@author: ljnolan

This is the specific implementation of BigWrapper for FABULOVS
"""
from supernolan import *
from affogato import automate, input_to_guess, get_flags, get_header_param, \
                       to_latex
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

def run_auto(c, side, control, psfkey, thorough, psfsub, radprof, bkp, custom):
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
      frame = 'icrs'
      wherecat = catalog[catalog['myID'] == c]
      pos = SkyCoord(ra=wherecat['ra']*u.degree, 
                     dec=wherecat['dec']*u.degree, frame=frame)
   size = [side, side]
   scale = [0.06, 0.06]
   # PSF Section
   if not control:
      psf = '../psf_automate/soi/ded_psf_combined.fits'
   else:
      if psfkey == 0:
         psf = '../psf_automate/soi/ded_psf_combined.fits'
      else:
         psf = '../psf_automate/%s/q%i_psf_combined.fits' % (name, psfkey)
         
   radfrac = 0.3
   
   backupdir = None
   if bkp:
      if not control:
         backupdir = '../backup/soi' #_dedPSF
      else:
         backupdir = '../backup/control'
         #if psfkey == 0:
         #   backupdir = '../backup/control_soiPSF'
         #else:
         #   backupdir = '../backup/control_selfPSF'
   
   if custom:
      customDir = 'custom_params'
      customParam = '%s.galfit.01' % name
      customCons = '%s.constraints' % name
   else:
      customDir = None
      customParam = None
      customCons = None
   
   automate(path, file, pos, size, scale, psf, backupdir=backupdir, 
            bkpname=name, customDir=customDir, customParam=customParam,
            customCons=customCons, useError=True, thorough=thorough,
            masking=True, psfsub=psfsub, radprof=radprof, flx=False,
            radfrac=radfrac)
   return


def make_table(control, skipbad=True):
   heads = ['Name', 'ID', 'Fit Chi2Nu', 'Component', 'Xpix', 'Ypix',
            'Mag', 'R_eff', 'Sersic', 'Axis Ratio', 'Pos. Angle']
   all_comps = []
   if not control:
      sample = range(9)[1:]
   else:
      sample = [2, 3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 44, 47, 48, 70, 80, 
                84, 85, 86, 87]
      useself = [19, 87] # 14 is also ok
      badagn = [5, 16, 19, 22, 33, 37, 47, 70, 80, 84, 86, 87]
   for c in sample:
      if not control:
         file = 'ie47%.2i' % c
         backupdir = 'backup/soi'
         catalog = Table.read('HST_resampled/catalog.csv', format='csv')
         name = catalog['name'][c-1]
         tabname = 'fit_soi.tab'
      else:
         if skipbad:
            if c in badagn:
               continue
         file = 'control%.3i' % c
         backupdir = 'backup/control'
         # TODO: add a check for which PSF...
         catalog = Table.read('Control_Sample/controlsample.csv', 
                              format='csv')
         name = catalog[catalog['myID'] == c]['SDSS Name'][0]
         tabname = 'fit_con.tab'
      flag = get_flags(file=file, fromdir=backupdir)
      chi2nu = get_header_param('CHI2NU', ext=2, file=file, fromdir=backupdir)
      comps = input_to_guess('%s/%s.galfit.01' % (backupdir, file))
      # Flip the order such that the central sersic is first, followed by
      # central PSF, followed by contaminant sersics, labelled as such
      if len(comps) > 2:
         comps = [comps[1], comps[0], *comps[2:]]
         for comp in comps[2:]:
            comp[0] = 'contaminant'
      else:
         comps = [comps[1], comps[0]]
      comps[1].insert(0, 'psf')
      for i, comp in enumerate(comps):
         comp.insert(0, c)
         if i == 0:
            comp.insert(0, name)
            comp.insert(2, chi2nu)
         else:
            comp.insert(0, '-99.0')
            comp.insert(2, -99.0)
         if flag == 1:
            for n, element in enumerate(comp):
               if n>3:
                  comp[n] = '*' + str(element)
         missing = len(heads) - len(comp)
         if missing > 0:
            comp += [-99.0] * missing
         all_comps.append(comp)
   tab = Table(names=heads, rows=all_comps)
   to_latex(tab, outputdir='backup', tabname=tabname)
   return
   

c = 5 # which file? "<class 'numpy.bool_'>"
startingk = 0 # midstart?

# Notes:
# Sample of Interest: 1-8
# 4 and 6 are bigger (at min 400x400)
# 3 is badly behaved and won't converge because of an overlapping guy,
# 5 & 7 also doesn't want to converge for no clear reason.

side = 400
control = False
compare = False
thorough = False
psfsub = True
radprof = True
loop = True
check = False
skipbad = True
custom = True
if control:
   sample = [2, 3, 5, 14, 16, 19, 22, 25, 33, 37, 40, 44, 47, 48, 70, 80, 84, 
             85, 86, 87]
   samplekeys = [0, 4, 2,  1,  4,  1,  0,  0,  2,  3,  3,  0,  3,  0,  4,  2,
                 2,  4,  2,  1]
   useself = [19, 87] # 14 is also ok
   badagn = [5, 16, 19, 22, 33, 37, 47, 70, 80, 84, 86, 87]
   customlist = [14, 25]
else:
   sample = [i+1 for i in range(8)]
   psfkey = 0
   customlist = [5]
if loop:
   bkp = True
   for k, b in enumerate(sample):
      if k < startingk:
         continue
      if b in customlist:
         custom = True
      else:
         custom = False
      sd = side
      if not control:
         if b in (4, 6):
            sd += 200
      else:
         if skipbad and (b in badagn):
            continue
         psfkey = 0
         if b in useself:
            psfkey = samplekeys[k]
      run_auto(b, sd, control, psfkey, thorough, psfsub, radprof, bkp, custom)
      if control and compare and psfkey != 0:
         run_auto(b, sd, control, 0, thorough, radprof, bkp, custom)
      if check and b != sample[-1]:
         ip = 'f'
         while ip not in ['y', 'n', 'Y', 'N']:
            print("Continue? [y/n]:")
            ip = input()
         if ip in ['n', 'N']:
            print('Your startingk value is ', k+1)
            break
else:
   if not control:
      if c in (4,6):
         side += 200
   psfkey = 0
   bkp = False
   run_auto(c, side, control, psfkey, thorough, psfsub, radprof, bkp, custom)
   
make_table(control)
