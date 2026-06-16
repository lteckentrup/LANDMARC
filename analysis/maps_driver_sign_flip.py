import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from  matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5

import argparse

'''
Initialise argument parsing: 
exp for LMT scenario (a6jn, a6xv,a6xx,a6xy)
first_year and last_year for fut. projection timeslice (2045-2059 or 2085-2099)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)

args = parser.parse_args()

### Assign variables
first_year=args.first_year
last_year=args.last_year

### Set pathway where input files are located
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Land ocean mask
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc').sel(lat=slice(-60, 90))

### Get gridarea
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc').sel(lat=slice(-60, 90))

### Align coordinates
ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']

### Mask ocean
da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf']))

### Define  TableIDs
ID_to_var_dict = {
    'Lmon': ['cVeg','cLitter'],
    'Emon': ['cLand', 'cSoil'],
    'Eyr': ['cBECCS', 'cBiochar', 'fco2fossub'],
    'Amon': ['tas', 'hfls', 'albedo'],
}
var_to_ID_dict = {var: id for id, vars in ID_to_var_dict.items() for var in vars}

### Find correct directory
directory_to_var_dict = {
    'carbon': ['cBECCS','cBiochar','fco2fossub','cLand','cSoil','cVeg','cLitter'],
    'climate': ['albedo','tas', 'pr', 'PET','hfls'],
    'drought': ['SPI_1', 'SPI_3', 'SPI_6', 'SPI_12',
                'SPEI_1', 'SPEI_3', 'SPEI_6', 'SPEI_12'],
    'fire': ['FFDI', 'FWI']
    }
var_to_directory_dict = {var: id for id, vars in directory_to_var_dict.items() for var in vars}

def custom_formatter(x, pos):
    if x == 0:
        return('0')
    else:
        return(f'{x:.1f}')

### Set up plot for maps
def make_map(var,first_year,last_year,exp_ref,exp_pert,position):
    global ds_mask
    global ds_gridarea
    
    ### Get TableID ID
    ID = var_to_ID_dict[var]
    directory = var_to_directory_dict[var]

    ### Read in masks where variable shows robust change
    da_robust = xr.open_dataset(pathwayIN+'robustness_maps/'+first_year+'-'+last_year+
                                '/'+directory+'/'+var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+exp_ref+'_'+
                                exp_pert+'_gr.nc').sel(lat=slice(-60, 90))['pvals'][0]

    ### Read in where tCDR could flip sign in variable change
    da_flip = xr.open_dataset(pathwayIN+'robustness_maps/'+first_year+'-'+last_year+
                            '/'+directory+'/'+var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+exp_ref+'_'+
                            exp_pert+'_gr_sign_flip.nc').sel(lat=slice(-60, 90))['flip']
    
    ### Align coordinates
    da_robust['lat'], da_robust['lon'] = ds_mask['lat'], ds_mask['lon']
    da_flip['lat'], da_flip['lon'] = ds_mask['lat'], ds_mask['lon']
    
    ### Mask ocean
    da_robust = da_robust.where(~np.isnan(ds_mask['sftlf']))
    da_flip = da_flip.where(~np.isnan(ds_mask['sftlf']))
    
    ### Keep mask where change in signs are associated with robust changes
    da_flip_sign = da_flip * da_robust
    
    ### Get percentage of landarea with sign. positive and negative values
    masks = {
        'pos': (da_flip_sign >= 0) & (da_robust == 1),
        'neg': (da_flip_sign <= 0) & (da_robust == 1)
    }

    ### Get percentage of global land
    percs = {
        key: 100 * ((da_gridarea * mask).sum(dim=['lat', 'lon']) / 
                    da_gridarea.sum(dim=['lat', 'lon']))
        for key, mask in masks.items()
    }

    ### Round results
    perc_sign = percs['pos'] + percs['neg']
    perc_sign_pos = percs['pos']
    perc_sign_neg = percs['neg']
    
    ### Print results sign. percentages in bottom left corner
    axs[position].text(
    0.02, 0.02, 
    f'Total: {perc_sign.item():.2f}%\n'
    f'Pos: {perc_sign_pos.item():.2f}%\n'
    f'Neg: {perc_sign_neg.item():.2f}%',
    transform=axs[position].transAxes,
    fontsize=8,
    color='black',
    bbox=dict(facecolor='white', alpha=0, edgecolor='none')
    )   
    
    ### Get latitude and longitude coordinates
    lat, lon = da_flip['lat'], da_flip['lon']
    
    ### Add cyclic point to da_flip_sign
    da_flip_cyclic, lon = add_cyclic_point(da_flip_sign, coord=da_flip.lon)

    ### Set colorbar: assign new colors to arrow tips so add two extra colors
    levels = [0,1,2]
    cols = ['#912a23', '#118e82']

    ### cmap based on number of levels
    cmap = ListedColormap(cols[:])
    
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=cmap.N)
    
    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da_flip_cyclic, cmap=cmap, norm=norm,zorder=1)
        
    # ### Mask ocean points    
    # da_robust = da_robust.where(~np.isnan(ds_mask['sftlf']))
    
    ### Set up stippling (hatching in python is fugly)
    ### Marker size and density of stippling
    marker_size = 0.04
    density = 8

    ### Find which gridcells are robust
    robust_points = da_robust == 0

    ### Grid mask for regular spacing
    grid_mask = np.zeros_like(robust_points.values, dtype=bool)
    grid_mask[::density, ::density] = True

    ### Combine robust points and grid mask
    robust_points_regular_spacing = robust_points & xr.DataArray(grid_mask,
                                                                 dims=robust_points.dims,
                                                                 coords=robust_points.coords)

    ### Coordinates of regularly spaced stippling points
    lat_robust, lon_robust = np.where(robust_points_regular_spacing)
    lon_robust_coords = da_robust.lon[lon_robust]
    lat_robust_coords = da_robust.lat[lat_robust]
    
    axs[position].scatter(lon_robust_coords,
                          lat_robust_coords,
                          marker=',',
                          s=marker_size,
                          color='k',
                          transform=ccrs.PlateCarree())

    ### Plot coastlines   
    axs[position].coastlines(linewidth=0.5)

    ### Drop spines
    axs[position].spines['geo'].set_visible(False)

    ### Reintroduce left and bottom spine
    axs[position].spines['left'].set_visible(True)
    axs[position].spines['bottom'].set_visible(True)
    
def call_plot():             
    ### Decide which carbon pools to plot
    cpool_1, cpool_2, cpool_3 = 'albedo','hfls','cLand'

    exp_ref_row_1, exp_pert_row_1  = 'a7cy', 'a7en'
    exp_ref_row_2, exp_pert_row_2 = 'a7cy', 'a7eo'
    exp_ref_row_3, exp_pert_row_3 = 'a7en', 'a7eo'

    ## Plot carbon pools
    make_map(cpool_1,first_year,last_year,exp_ref_row_1,exp_pert_row_1,0)
    make_map(cpool_2,first_year,last_year,exp_ref_row_1,exp_pert_row_1,3)
    make_map(cpool_3,first_year,last_year,exp_ref_row_1,exp_pert_row_1,6)
    make_map(cpool_1,first_year,last_year,exp_ref_row_2,exp_pert_row_2,1)
    make_map(cpool_2,first_year,last_year,exp_ref_row_2,exp_pert_row_2,4)
    make_map(cpool_3,first_year,last_year,exp_ref_row_2,exp_pert_row_2,7)
    make_map(cpool_1,first_year,last_year,exp_ref_row_3,exp_pert_row_3,2)
    make_map(cpool_2,first_year,last_year,exp_ref_row_3,exp_pert_row_3,5)
    make_map(cpool_3,first_year,last_year,exp_ref_row_3,exp_pert_row_3,8)

    ## Plot subplot titles
    axs[0].set_title('Moderate CDR Ambition vs. Reference (2071-2100)\nAlbedo',
                     fontsize=9,linespacing=1.75)
    axs[3].set_title('Latent heat flux',fontsize=10)
    axs[6].set_title('Carbon stored in land',fontsize=10)
    
    axs[1].set_title('High CDR Ambition vs. Reference (2071-2100)\nAlbedo',
                     fontsize=9,linespacing=1.75)
    axs[4].set_title('Latent heat flux',fontsize=10)
    axs[7].set_title('Carbon stored in land',fontsize=10)

    axs[2].set_title('High vs. Moderate CDR Ambition (2071-2100)\nAlbedo',
                     fontsize=9,linespacing=1.75)                        
    axs[5].set_title('Latent heat flux',fontsize=10)
    axs[8].set_title('Carbon stored in land',fontsize=10)

    ### Set up positions and labels
    positions=[0,1,2,
               3,4,5,
               6,7,8]
    title_index=['a)','b)','c)',
                 'd)','e)','f)',
                 'g)','h)','i)']

    ### Loop through plot command, and adjust subplot titles
    for p, ti in zip(positions, title_index):
        axs[p].set_title(ti, fontsize=9, loc='left')

    ### Adjust ticklabels on axes
    for p in (0,3,6):
        ### Show ticklabels left
        axs[p].yaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelsize=8)

    for p in (1,2,4,5,7,8):
        axs[p].yaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelleft=False)
        
    for p in (0,1,2,3,4,5):
        axs[p].xaxis.set_visible(True)
        axs[p].tick_params(axis='x', labelbottom=False)

    for p in (6,7,8):
        ### Show ticklabels bottom
        axs[p].xaxis.set_visible(True)
        axs[p].tick_params(axis='x', labelsize=8)

    plt.subplots_adjust(**plot_params)
    plt.savefig('figures/maps_driver_'+first_year+'-'+last_year+'_flip_sign.png',dpi=400)

### Set up figure    
figsize=(14,7)
plot_params = {'top':0.96, 'left':0.03,
               'right':0.96, 'bottom':0.05,
               'wspace':0.05, 'hspace':0.05}
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize,
                        sharex=True, sharey=True)
axs=axs.flatten()

### Manual legend
fig.add_artist(
    Rectangle(
        (0.32, 0.01),   # x, y in figure coordinates
        0.05,           # width
        0.03,           # height
        transform=fig.transFigure,
        facecolor='#912a23',
        edgecolor='none'
    )
)

fig.add_artist(
    Rectangle(
        (0.5, 0.01),
        0.05,
        0.03,
        transform=fig.transFigure,
        facecolor='#118e82',
        edgecolor='none'
    )
)

### Add labels
fig.text(0.38, 0.025, r'Decrease $\rightarrow$ Increase', ha='left', 
         va='center', fontsize=10, color='#912a23')
fig.text(0.67, 0.025, r'Increase $\rightarrow$ Decrease', ha='right', 
         va='center', fontsize=10, color='#118e82')

call_plot()
