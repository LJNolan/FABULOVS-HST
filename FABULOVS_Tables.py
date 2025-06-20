#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:05:37 2024

@author: ljnolan

Table generation for FABULOVS
"""
from supernolan import *
from affogato import from_latex, to_latex
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, vstack
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
plt.rcParams.update({'font.size': 18})

def composed_table(cols=None, comps=False, soi=True, control=True):
   backupdir = 'backup'
   if soi:
      # Generate large SoI Table
      soi_fit_tab = from_latex(backupdir, 'fit_soi.tab')
      soi_sta_tab = from_latex(backupdir, 'stats_soi.tab')
      soi_sta_tab.remove_column('Name')
      soi_mega_tab = join(soi_fit_tab, soi_sta_tab, keys='ID', 
                          join_type='outer')
      
      # Sort table and mask appropriately to get aesthetically pleasing blanks
      soi_mega_tab.sort('ID')
      soi_mega_tab = soi_mega_tab.group_by('ID')
      for group in soi_mega_tab.groups:
         group.sort('Component', reverse=True)
      soi_mega_tab.add_column(['SoI']*len(soi_mega_tab), name='Sample', 
                              index=2)
      soi_mega_tab= Table(soi_mega_tab, masked=True, copy=False)
      for cn in ['Sample', *soi_sta_tab.colnames]:
         soi_mega_tab[cn].mask = soi_mega_tab['Name'].mask
   if control:
      # Generate large Control Table
      con_fit_tab = from_latex(backupdir, 'fit_con.tab')
      con_sta_tab = from_latex(backupdir, 'stats_con.tab')
      con_sta_tab.remove_column('Name')
      con_mega_tab = join(con_fit_tab, con_sta_tab, keys='ID', 
                          join_type='outer')
      
      # Sort table and mask appropriately
      con_mega_tab.sort('ID')
      con_mega_tab = con_mega_tab.group_by('ID')
      for group in con_mega_tab.groups:
         group.sort('Component', reverse=True)
      con_mega_tab.add_column(['Control']*len(con_mega_tab), name='Sample',
                              index=2)
      con_mega_tab= Table(con_mega_tab, masked=True, copy=False)
      for cn in ['Sample', *con_sta_tab.colnames]:
         con_mega_tab[cn].mask = con_mega_tab['Name'].mask
   
   # Stack if both SoI and control are requested
   if soi and control:
      mega_tab = vstack([soi_mega_tab, con_mega_tab])
   elif soi:
      mega_tab = Table(soi_mega_tab, masked=True, copy=True)
   elif control:
      mega_tab = Table(con_mega_tab, masked=True, copy=True)
   else:
      print("What are you doing?  'soi' and 'control' are both False!")
      return
   
   # For some reason this produces some blank rows.  Clean:
   badrows = []
   for i, row in enumerate(mega_tab):
      if mega_tab['Component'].mask[i]:
         badrows.append(i)
   mega_tab.remove_rows(badrows)
   
   # Reformat names
   namefix = ['Name', 'ID', 'Sample', r'Fit $\chi^{2}/\nu$',
              'Component', r'$X_{pix}$', r'$Y_{pix}$', 'Mag', r'$R_{eff} [pix]$',
              r'Sérsic index', 'Axis Ratio', 'Pos. Angle', r'$A_{net}$',
              r'$A_{S, net}$', r'S(G, M$_{20}$)$_{net}$',
              r'F(G, M$_{20}$)$_{net}$', r'$A_{sub}$', r'$A_{S, sub}$',
              r'S(G, M$_{20}$)$_{sub}$', r'F(G, M$_{20}$)$_{sub}$']
   for i, name in enumerate(mega_tab.colnames):
      mega_tab.rename_column(name, namefix[i])
   
   # Rounding
   num_names = [r'Fit $\chi^{2}/\nu$', r'$X_{pix}$', r'$Y_{pix}$',
                'Mag', r'$R_{eff} [pix]$', r'Sérsic index', 'Axis Ratio',
                'Pos. Angle', r'$A_{net}$', r'$A_{S, net}$',
                r'S(G, M$_{20}$)$_{net}$', r'F(G, M$_{20}$)$_{net}$',
                r'$A_{sub}$', r'$A_{S, sub}$', r'S(G, M$_{20}$)$_{sub}$',
                r'F(G, M$_{20}$)$_{sub}$']
   decimals = [2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2]
   mega_tab = round_table_dec(mega_tab, num_names, decimals)
   
   # Generate other tables
   model = ['Name', 'Sample', r'Fit $\chi^{2}/\nu$',
            'Component',# r'$X_{pix}$', r'$Y_{pix}$',
            'Mag', r'$R_{eff} [pix]$', r'Sérsic index',
            'Axis Ratio', 'Pos. Angle']
   model_tab = mega_tab[model].copy()
   
   stats = ['Name', 'Sample', r'Fit $\chi^{2}/\nu$', r'Sérsic index',
            r'$A_{net}$', r'$A_{S, net}$', r'S(G, M$_{20}$)$_{net}$',
            r'F(G, M$_{20}$)$_{net}$', r'$A_{sub}$', r'$A_{S, sub}$',
            r'S(G, M$_{20}$)$_{sub}$', r'F(G, M$_{20}$)$_{sub}$']
   stats_tab = mega_tab[stats].copy()[~mega_tab['Name'].mask]
      
   if cols is None:
      custom_tab = None
      return model_tab, stats_tab
   else:
      if not comps:
         custom_tab = mega_tab[cols].copy()[~mega_tab['Name'].mask]
      else:
         custom_tab = mega_tab[cols].copy()
   return model_tab, stats_tab, custom_tab


def custom_latex(tabname='fit.tab', inputdir='.', caption=None, label=None):
   tabname = '%s/%s' % (inputdir, tabname)
   with open(tabname, 'r') as file:
      data = file.readlines()
   
   data[0] = '\\begin{table*}\n'
   data[-1] = '\\end{table*}\n'
   m = 1
   if label is not None:
      data.insert(1, '\\label{'+label+'}\n')
      m += 1
   if caption is not None:
      data.insert(1, '\\caption{'+caption+'}\n')
      m += 1
   data.insert(1, '\\centering\n')
   m0 = m
   data[1+m] = data[1+m].replace(r'{c', r'{ c').replace(r'c}', r'c }')
   for i, line in enumerate(data[2+m:-2]):
      if line.split()[0] != r'&':
         data.insert(2+m+i, '\\hline\n')
         m+=1
   for i, line in enumerate(data[2+m0+1:5+m0]):
      if line.split()[0] =='\\hline':
         data[2+m0+i+1] = '\\hline\\hline\n'
   
   with open(tabname, 'w') as file:
      file.writelines(data)
   return


def multihist(x, hue, n_bins=10, color=None, colors=None, hatchs=None, **kws):
   """
   Wrapper for sns.histplot() to match bin sizes between hues.

   Parameters
   ----------
   x : pandas dataframe
      DESCRIPTION.
   hue : str
      Column of ``x`` by which to separate data, i.e. sns.FacetGrid hue.
   n_bins : int, optional
      Number of bins. The default is 10.
   color : string, optional
      Needed to be passed to histplot. The default is None.
   colors : dict, optional
      Dict mapping ``hue`` types to colors. The default is None.
   **kws : keyword arguments
      Passed to histplot.

   Returns
   -------
   None.

   """
   bins = np.linspace(x.min(), x.max(), n_bins)
   for key, x_i in x.groupby(hue):
      if colors is None and hatchs is None:
         sns.histplot(x_i, bins=bins, label=key, **kws)
      elif hatchs is None:
         sns.histplot(x_i, bins=bins, label=key, color=colors[key], **kws)
      elif colors is None:
         sns.histplot(x_i, bins=bins, label=key, hatch=hatchs[key], **kws)
      else:
         sns.histplot(x_i, bins=bins, label=key, color=colors[key],
                      hatch=hatchs[key], **kws)
   return


def multibox(x, hue, color=None, colors=None, hatchs=None, **kws):
   """
   Wrapper for sns.boxplot() to match multihist above - this isn't really
   useful unless you add something to customize the boxes...

   Parameters
   ----------
   x : pandas dataframe
      DESCRIPTION.
   hue : str
      Column of ``x`` by which to separate data, i.e. sns.FacetGrid hue.
   n_bins : int, optional
      Number of bins. The default is 10.
   color : string, optional
      Needed to be passed to histplot. The default is None.
   colors : dict, optional
      Dict mapping ``hue`` types to colors. The default is None.
   **kws : keyword arguments
      Passed to histplot.

   Returns
   -------
   None.

   """
   if hatchs is None:
      order = None
   else:
      order=list(hatchs.keys())
   bp = sns.boxplot(x=x, hue=hue, hue_order=order, palette=colors, **kws)
   if hatchs is None:
      pass
   else:
      for i, cont in enumerate(bp.containers):
         box = cont.boxes[0]
         box.set_hatch(hatchs[order[i]])
         # A little trickiness is required here to make hatching appear on
         # legends.
         # Hint: bp.get_legend().legend_handles[i].set_hatch(hatchs[order[i]])
   
   return bp


def fixed_boxplot(x, y, *args, label=None, **kwargs):
    return sns.boxplot(x=x, y=y, *args, **kwargs, boxprops={'label' : label})


def doit(what):
   g = sns.FacetGrid(stats_df2, col='variable', col_wrap=3, sharex=False,
                     sharey=False, height=5, aspect=0.8)
   if what == 'hist':
      g.map(multihist, 'value', sdf_cols[0], colors=color_map, hatchs=kind_map,
            edgecolor='k')
      g.set_ylabels('Count')
   elif what == 'box':
      g.map(multibox, 'value', sdf_cols[0], colors=color_map, hatchs=kind_map)
      g.set_ylabels('')
   g.set_titles('{col_name}')
   for ax in g.axes.flat:
       ax.set_title(ax.get_title(), y=-0.12, va='top')
   g.set_xlabels('')
   g.add_legend(loc='upper center', bbox_to_anchor=(0.6, 0.65))
# =============================================================================
# Non-functional attempt to make hatching appear in boxplot legend
#    if what == 'box':
#       for i, legtext in enumerate(g.fig.legend().get_texts()[:len(kind_map)]):
#          sample = legtext.get_text()
#          print(sample)
#          g.fig.legend().legend_handles[i].set_hatch(kind_map[sample])
# =============================================================================
   g.fig.tight_layout()
   #plt.show()
   plt.savefig('misc/stats_%s.png' % what, bbox_inches='tight')
   return


def bip(bop):
   return np.array(list(bop))


def ref_tab():
   # Import and format targets and control
   targets = pd.read_csv('Control_sample/8targets.csv')
   targets.rename(columns={'name':'Name', 'ra':'RA', 'dec':'DEC'}, inplace=True)
   targets['Sample'] = ['SoI'] * 8
   
   controls = pd.read_csv('Control_Sample/control.csv')
   controls.rename(columns={'Target Name':'Name', 'ra':'RA', 'dec':'DEC'}, inplace=True)
   controls['Sample'] = ['Control'] * 8
   
   # Merge tables
   frames = [targets, controls]
   combined = pd.concat(frames,join='inner', ignore_index=True)
   combined = combined[['Name', 'Sample', 'z', 'RA', 'DEC']]
   
   # Import other data from SDSS DR7
   dr7 = fits_to_pd('misc/dr7_bh_Nov19_2013.fits')
   addl_cols = ['MI_Z2', 'LOGBH_HB_MD04']
   combined = match_and_append(combined, dr7, addl_cols)
   
   # Formatting
   combined = combined.rename(columns={'MI_Z2':r'$M_i$', 
                           'LOGBH_HB_MD04':r'Virial Mass [log10($M_{\odot}$)]'})
   combined = combined.round({'RA': 4, 'DEC': 4, r'$M_i$': 3,
                              r'Virial Mass [log10($M_{BH}$)]': 3})
   
   # Convert to astropy
   tab = Table.from_pandas(combined)
   return tab


def match_and_append(table1, table2, columns_to_append):
   # Create SkyCoord objects from RA/DEC columns (assumed in degrees)
   coords1 = SkyCoord(ra=table1['RA'].values*u.deg,
                      dec=table1['DEC'].values*u.deg)
   coords2 = SkyCoord(ra=table2['RA'].values*u.deg,
                      dec=table2['DEC'].values*u.deg)
   
   # Match each coord in table1 to the closest in table2
   idx, _, _ = coords1.match_to_catalog_sky(coords2)
   
   # Select and append the desired columns from table2
   matched_data = table2.iloc[idx][columns_to_append].reset_index(drop=True)
   return pd.concat([table1.reset_index(drop=True), matched_data], axis=1)


def fits_to_pd(file):
   # Only allows single-dimensional columns
   with fits.open(file) as hdul:
      table_data = hdul[1].data
      tab = Table(table_data)
      names = [name for name in tab.colnames if len(tab[name].shape) <= 1]
      df = tab[names].to_pandas()
   return df


# Color blindness friendly cycle - good contrast between subsequent colors
# Credit to thivest on GitHub
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
CBcc = CB_color_cycle # shortname

# === Specifications ===
backupdir = 'backup'
cols = []

name1 = 'model.tab'
caption1 = '''
\\galfit\\ model parameters and associated data, for the SoI and control
 sample.  Values above the following thresholds indicate mergers:
 $A\\geq0.35$, $A_S\\geq0.4$, S(G, M$_{20}$)$\\geq0$ 
 \\citep{wilkinson_merger_2022}.
'''
label1 = 'tab:cat'

name2 = 'stats.tab'
caption2= '''
Merger statistics for the SoI and control sample; $A_{net}$, $A_{S, net}$,
S(G, M$_{20}$)$_{net}$, and F(G, M$_{20}$)$_{net}$ are the asymmetry, shape
asymmetry, and Gini-M20 metrics (merger and bulge, respectively)
as described in Section \\ref{sec:met} for the full science image, and those
with the -sub prefix are for the science image minus the central PSF model.
\\Sersic\\ indices from poorly constrained fits, as above, are marked with an
asterisk(*).
'''
label2 = 'tab:met'

name3 = 'average.tab'
caption3 = '''
Merger statistic averages for the SoI and control sample, followed by the
difference (Control - SoI).
'''
label3 = 'tab:avg'

name4 = 'metastat.tab'
caption4 = '''
Statistics ($S$) and p-values ($p$) from the Kolmogorov-Smirnov ($KS$) and
Mann-Whitney U ($MWU$) tests run on merger statistics.  A p-value of
$\\leq 0.003$ indicates our SoI and control come from different underlying
populations at roughly $3\\sigma$ confidence.
'''
label4 = 'tab:mst'

# === Run ===
model_tab, stats_tab = composed_table()

# Compute Averages
avg_cols = [r'Sérsic index', r'$A_{net}$', r'$A_{S, net}$',
            r'S(G, M$_{20}$)$_{net}$', r'F(G, M$_{20}$)$_{net}$', r'$A_{sub}$',
            r'$A_{S, sub}$', r'S(G, M$_{20}$)$_{sub}$',
            r'F(G, M$_{20}$)$_{sub}$']
avgs_soi, avgs_con = [], []
for col in avg_cols:
   avgs_soi.append(np.mean(stats_tab[stats_tab['Sample'] == 'SoI'][col].data))
   avgs_con.append(np.mean(stats_tab[stats_tab['Sample'] == 'Control'][col].data))
avgs = np.array([avgs_soi, avgs_con])
avg_tab = Table(data=avgs, names=avg_cols)
avg_tab.add_row(bip(avg_tab[1]) - bip(avg_tab[0]))
avg_tab = round_table_dec(avg_tab, avg_cols, 2)
avg_tab['Sample'] = ['SoI', 'Control', r'$\Delta_{c-s}$']
avg_tab = avg_tab['Sample', *avg_cols]

# Generate histograms of statistics
stats_df = stats_tab['Sample', *avg_cols[1:], avg_cols[0]].to_pandas()
#avg_cols[0] = r'S$\acute{e}$rsic'
stats_df.rename(columns={r'\Sersic':r'S$\acute{e}$rsic'}, inplace=True)

sdf_cols = stats_df.columns.tolist()
samples = stats_df[sdf_cols[0]].unique()
color_map, kind_map = style_dict(samples, kind='hatch', match=True)
stats_df2 = pd.melt(stats_df, id_vars=sdf_cols[0],
                    value_vars=sdf_cols[1:],
                    value_name='value')

doit('hist')
doit('box')

# Generate meta-statistics
metastats = []
for col in avg_cols:
   s1 = stats_tab[stats_tab['Sample'] == 'SoI'][col].data
   s2 = stats_tab[stats_tab['Sample'] == 'Control'][col].data
   res1 = stats.kstest(s1, s2)
   res2 = stats.mannwhitneyu(s1, s2)
   metastats.append([res1.statistic, res1.pvalue, res2.statistic, res2.pvalue])
   # KS val, KS p, M-W U val, M-W U p
metastats = np.round(np.array(metastats).T, decimals=3)
mstat_tab = Table(metastats, names=avg_cols)
self_rem = '-99.0'
mstat_tab[self_rem] = [r'$S_{KS}$', r'$p_{KS}$', r'$S_{MWU}$', r'$p_{MWU}$']
mstat_tab = mstat_tab[self_rem, *avg_cols]

# Generate and customize latex tables for paper
tabs = [model_tab, stats_tab, avg_tab, mstat_tab]
tabnames = [name1, name2, name3, name4]
delvals = [None, None, None, self_rem]
captions = [caption1, caption2, caption3, caption4]
labels = [label1, label2, label3, label4]

for n, tab in enumerate(tabs):
   to_latex(tab, outputdir=backupdir, tabname=tabnames[n], delval=delvals[n])
   custom_latex(tabname=tabnames[n], inputdir=backupdir, caption=captions[n],
                label=labels[n])

ref = ref_tab()
tabname = 'ref.tab'
caption = '''
Names and coordinates of AGN hosts from sample of interest (SoI)
and control sample.
'''
label = 'tab:ref'
to_latex(ref, outputdir=backupdir, tabname=tabname)
custom_latex(tabname=tabname, inputdir=backupdir, caption=caption,
             label=label)
