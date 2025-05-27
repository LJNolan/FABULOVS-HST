#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:58:05 2023

@author: ljnolan

Code to drizzle together the images pulled from MAST for the control sample.

Note:  I may have removed the HST files this code references when cleaning up
directories.
"""
import drizzlepac
from stwcs import updatewcs
import os


path = 'Control_Sample/HST_Files2/control%.3i/' # WAS HST_Files2
os.environ['iref'] = '/Users/ljnolan/Documents/BSBH_2022/' + \
                     'Control_Sample/HST_Files/reffiles/'
#os.system('export iref=/Users/ljnolan/Documents/BSBH_2022/' + 
#          'Shreya_Control_Sample/HST_Files/reffiles/')
#???

# Control Sample

#for i in range(2,3):
   #print(path % i)
   #if i == 6 : pixfrac = 0.95
   #elif i == 7 : pixfrac = 0.95
   #else: pixfrac = 0.8
   #pixfrac = 0.8
# 3, 5, 14, 25, 33, 37, 44, 80, 84, 85, 86
# 2, 16, 19, 22, 40, 47, 48, 70, 87
# Good ones: 2, 3, 14, 25, 40, 44, 48, 85
i = 85 # Fails: 
pixfrac = 0.8
updatewcs.updatewcs((path + '*flt.fits') % i)
drizzlepac.astrodrizzle.AstroDrizzle((path + '*flt.fits') % i,
                                     output=(path + 'control%.3i') % (i, i),
                                     final_pixfrac=pixfrac,
                                     final_wht_type='ERR',
                                     configobj='input.cfg')#,
#                                     updatewcs='Yes')
