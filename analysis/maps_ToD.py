import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from  matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--var_left', type=str, required=True)
parser.add_argument('--var_right', type=str, required=True)

args = parser.parse_args()

### Assign variables
var_left=args.var_left
var_right=args.var_right

### Set pathway where input files are located
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc').sel(lat=slice(-60, 90))

### Get gridarea
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc').sel(lat=slice(-60, 90))
ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']
da_gridarea = ds_gridarea['cell_area'].where(ds_mask['sftlf'] != 0, np.nan)

def get_data(var,ref,exp_pert):
    global pathwayIN
    global ds_mask
    
    ds = xr.open_dataset(pathwayIN+'/ToD/2071-2100/'+var+'/ToD_'+
                         var+'_'+ref+'_'+exp_pert+'_0.8.nc').sel(lat=slice(-60, 90))
    ds['lat'], ds['lon'] = ds_gridarea['lat'], ds_gridarea['lon'
                                                           ]
    ### Set ocean pixels to nan
    da = ds['ToD'].where(ds_mask['sftlf'] != 0, np.nan)    
    return(da)

### Set up plot for maps
def make_map(var,ref,exp_pert,position):  
    ### Get data
    da = get_data(var,ref,exp_pert)
        
    ### Get latitude and longitude coordinates
    lat, lon = da.lat, da.lon
            
    ### Set levels for colorbar
    levels = np.arange(-2095,-2015,5).tolist() + np.arange(2020,2100,5).tolist()
    
    ### Define colormaps: Get hexacodes for split cmap
    colors_increase = sns.color_palette('inferno', 16)
    colors_decrease = sns.color_palette('PuBuGn', 16) 

    ### Set up colormaps: combined and separate
    cmap_combined = mcolors.ListedColormap(colors_decrease+colors_increase)
    cmap_increase= mcolors.ListedColormap(colors_increase)
    cmap_decrease = mcolors.ListedColormap(colors_decrease)

    ### Set up bounds and normalisation
    bounds=levels
    norm = BoundaryNorm(bounds, ncolors=cmap_combined.N)
        
    ### Add cyclic point
    da, lon = add_cyclic_point(da, coord=lon)

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap_combined, norm=norm,zorder=1)
    
    ### Plot coastlines
    axs[position].coastlines(linewidth=0.5)

    ### Drop spines
    axs[position].spines['geo'].set_visible(False)

    ### Reintroduce left and bottom spine
    axs[position].spines['left'].set_visible(True)
    axs[position].spines['bottom'].set_visible(True)

    ### Cut out Antarctica
    coords = [-180,180,-60,90]
            
    ### Set extent of map
    axs[position].set_extent(coords, 
                             crs=ccrs.PlateCarree())

    ### Ensure same latitude tick spacing as in other plots
    axs[position].set_yticks([-50, -25, 0, 25, 50, 75], crs=ccrs.PlateCarree())
    axs[position].set_yticklabels(['-50', '-25', '0', '25', '50', '75'])

    ### Plot colormaps
    if position == 4:
        ### Set up bounds for 'mock' cmap
        mock_bounds = np.arange(2020, 2100, 5)

        ### Define normalisation
        mock_norm = mcolors.BoundaryNorm(mock_bounds, ncolors=len(mock_bounds), extend='neither')

        ### Set up separate colormaps
        cm_increase = cm.ScalarMappable(norm=mock_norm, cmap=cmap_increase)
        cm_decrease = cm.ScalarMappable(norm=mock_norm, cmap=cmap_decrease.reversed())

        ### Add both colorbars: increase at the top, decrease at the bottom
        cbax_increase = fig.add_axes([0.05, 0.1, 0.9, 0.025])
        cbax_decrease = fig.add_axes([0.05, 0.06, 0.9, 0.025])
        cbax_increase = plt.colorbar(cm_increase, cax=cbax_increase, orientation='horizontal')
        cbar_decrease = plt.colorbar(cm_decrease, cax=cbax_decrease, orientation='horizontal')

        ### Set ticks at edges of colorbar segments, plot ticks for increase at top and for decrease at bottom
        cbax_increase.set_ticks(mock_bounds)
        cbax_increase.set_ticklabels([str(year) for year in mock_bounds])
        cbax_increase.ax.xaxis.set_ticks_position('top')
        cbax_increase.ax.xaxis.set_label_position('top')

        cbar_decrease.set_ticks(mock_bounds)
        cbar_decrease.set_ticklabels([str(year) for year in mock_bounds])

figsize=(8.4,7.5)
plot_params = {'top':0.95, 'left':0.05,
               'right':0.97, 'bottom':0.2,
               'wspace':0.05, 'hspace':0}

fig, axs = plt.subplots(nrows=3,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)
axs=axs.flatten()

make_map(var_left,'a7cy','a7cy',0)
make_map(var_right,'a7cy','a7cy',1)
make_map(var_left,'a7cy','a7en',2)
make_map(var_right,'a7cy','a7en',3)
make_map(var_left,'a7en','a7eo',4)
make_map(var_right,'a7en','a7eo',5)

if var_left == 'tas':
    title_var_left = 'Temperature'
elif var_left == 'tx90p':
    title_var_left = 'TX90p'
if var_right == 'pr':
    title_var_right = 'Precipitation'
elif var_right == 'dry_days':
    title_var_right = 'Dry days'
    
axs[0].set_title(title_var_left+'\nReference (2015-2100 vs. 1981-2010)',
                 fontsize=9,linespacing=1.75)
axs[1].set_title(title_var_right+'\nReference (2015-2100 vs. 1981-2010)',
                 fontsize=9,linespacing=1.75)
axs[2].set_title('Moderate CDR Ambition vs. Reference (2015-2100)',fontsize=9)
axs[3].set_title('Moderate CDR Ambition vs. Reference (2015-2100)',fontsize=9)
axs[4].set_title('High vs. Moderate CDR Ambition (2015-2100)',fontsize=9)
axs[5].set_title('High vs. Moderate CDR Ambition (2015-2100)',fontsize=9)

positions=[0,1,
           2,3,
           4,5]
title_index=['a)','b)',
             'c)','d)',
             'e)','f)']

### Loop through plot command, and adjust subplot titles
for p, ti in zip(positions, title_index):
    axs[p].set_title(ti, loc='left',fontsize=9)

### Adjust ticklabels on axes
for p in (0,2,4):
    ### Show ticklabels left
    axs[p].yaxis.set_visible(True)

for p in (4,5):
    ### Show ticklabels bottom
    axs[p].xaxis.set_visible(True)

for p in (1,3,5):
    axs[p].yaxis.set_visible(True)
    axs[p].tick_params(axis='y', labelleft=False)

### Define again colors
colors_increase = sns.color_palette('inferno', 16)
colors_decrease = sns.color_palette('PuBuGn', 16)   

### Include annotations 
fig.text(0.05, 0.015, 'Early onset cooling/ wetting', ha='left', 
         va='center', fontsize=10, color=colors_decrease[::-1][0])
fig.text(0.95, 0.015, 'Late onset cooling /wetting', ha='right', 
         va='center', fontsize=10, color=colors_decrease[::-1][-8])
fig.text(0.05, 0.17, 'Early onset warming/ drying', ha='left', 
         va='center', fontsize=10, color=colors_increase[0])
fig.text(0.95, 0.17, 'Late onset warming/ drying', ha='right', 
         va='center', fontsize=10, color=colors_increase[-8])

plt.subplots_adjust(**plot_params)
plt.savefig('figures/climate/maps_ToD_'+var_left+'_'+var_right+'.png',dpi=400)
