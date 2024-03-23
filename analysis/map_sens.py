import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from  matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.util import add_cyclic_point
import matplotlib.patches as mpatches

import argparse

'''
Initialise argument parsing: 
exp for LMT scenario (a6jn, a6xv,a6xx,a6xy)
first_year and last_year for fut. projection timeslice (2045-2059 or 2085-2099)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--exp', type=str, required=False)
parser.add_argument('--ref', type=str, required=False)
parser.add_argument('--plot_type', type=str, required=False)

args = parser.parse_args()

### Assign variables
first_year=args.first_year
last_year=args.last_year
region=args.region

### Set pathway where input files are located
pathwayIN='/esarchive/scratch/lteckent/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'gridarea/landmask.nc')
    
def get_data(var,first_year,last_year,exp):
    global pathwayIN
    global ds_mask
    
    if var == 'cVeg':
        ID = 'Lyr'
    elif var in ('cBECCS','cBiochar','fco2fossub','cLand','cSoil'):
        ID = 'Eyr'
    elif var in ('tas','pr'):
        ID = 'Ayr'
        
    ### Read in variable
    if var in ('pasture', 'cropland', 'natural'):
        ### Read in all natural vegetation covers
        fname = pathwayIN+exp+'/lu_frac_'+exp+'_years.nc'       
    else:
        fname = pathwayIN+exp+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_r1i1p1f1_gr_2015-2100.nc'
        
    ds = xr.open_dataset(fname).sel(time=slice(first_year,
                                               last_year)).mean(dim='time')

    ### Set latitudes and longitudes (? precision in a6xx slightly off maybe?)
    ds['lat'] = ds_mask['lat']
    ds['lon'] = ds_mask['lon']
          
    ### Mask ocean points
    da = ds[var].where(ds_mask['sftlf'] != 0, np.nan)
    
    return(da) 

### Set up plot for maps
def make_map(var,first_year,last_year,exp,ref,position,region):
    ### Get diff
    diff_raw = get_data(var,first_year,last_year,exp) -  \
               get_data(var,first_year,last_year,ref)

    if region in ('global','ACTO'):
        diff = diff_raw
    else:
        ds_region = xr.open_dataset(pathwayIN+'/data/region_shape/mask_'+region+'.nc')
        ds_region_invertlat = ds_region.reindex(lat=list(reversed(ds_region['lat'])))
        var_mask = diff_raw.values * ds_region_invertlat['lsm'][0].values
        diff = xr.DataArray(var_mask, 
                            coords=diff_raw.coords, 
                            dims=diff_raw.dims)
            
    ### Set levels for colorbar
    if var in ('cropland','natural','pasture'):
        levels = [-100,-50,-20,-10,-5,-2,-1,1,2,5,10,20,50,100]
        cmap='BrBG'
    elif var == 'tas':
        levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5]
        cmap='YlGnBu_r'
    elif var == 'pr':
        levels = [-500,-200,-100,-50,-20,-10,-5,5,10,20,50,100,200,500]
        cmap='BrBG'
    else:
        levels = [-5000,-2000,-1000,-500,-200,-100,-50,-20,-10,-5,
                  5,10,20,50,100,200,500,1000,2000,5000]
        cmap='BrBG'
        diff = diff * 1000

    ### Set colorbar: assign new colors to arrow tips so add two extra colors
    pal = sns.color_palette(cmap, len(levels)+2)
    cols = pal.as_hex()
    
    if var == 'tas':
        pal = sns.color_palette('magma_r', len(levels))
        cols_tas = pal.as_hex()

        cols[-2] = cols_tas[1]
        cols[-4] = cols_tas[0]

        ### cmap based on actual number of levels (exclude first and last cmap val)
        cmap = ListedColormap(cols[1:-1])

        ### Set arrow color to first and last cmap val
        cmap.set_over(cols_tas[2])
        cmap.set_under(cols[0])

    else:
        ### Set low values to grey
        cols[int(len(levels)/2)] = '#d3d3d3'

        ### cmap based on actual number of levels (exclude first and last cmap val)
        cmap = ListedColormap(cols[1:-1])

        ### Set arrow color to first and last cmap val
        cmap.set_under(cols[0])
        cmap.set_over(cols[-1])
        
    ### Get latitude and longitude coordinates
    lat, lon = diff.lat, diff.lon
    
    if region == 'ACTO':
        pass
    else:
        ### Add cyclic point
        diff, lon = add_cyclic_point(diff, coord=lon)

    ### Set projection
    projection = ccrs.PlateCarree()   

    ### Boundary norm: set ncolors to ACTUAL number of levels (excl. first and last cmap val)
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=len(cols[1:-1]))

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, diff, cmap=cmap, norm=norm,zorder=1)
    axs[position].coastlines()

    ### Drop spines
    axs[position].spines['geo'].set_visible(False)

    ### Reintroduce left and bottom spine
    axs[position].spines['left'].set_visible(True)
    axs[position].spines['bottom'].set_visible(True)

    ### Cut out Antarctica
    if region == 'global':
        coords = [-180,180,-60,90]
    elif region == 'EAC':
        coords = [14,43,-13.5,13]
    elif region == 'EU27':
        coords = [-10,32,33,71]
    elif region == 'ASEAN':
        coords = [90,150,-12,30]
    elif region == 'NAMERICA':
        coords = [-168,-52,5,85]
    elif region == 'ACTO':
        coords = [-90,-30,-35,12]
        
        ### Plot contour of ACTO region instead of masking
        ds_region_invertlat = xr.open_dataset(pathwayIN+
                                              '/data/region_shape/mask_'+region+'.nc')
        ### Invert latitude
        ds_region = ds_region_invertlat.reindex(lat=list(reversed(ds_region_invertlat['lat'])))
        
        ### Plot contour of ACTO on top of variable
        axs[position].contour(lon, lat, ds_region_invertlat['lsm'][0].fillna(0), 
                              colors='tab:red',linewidths=0.5,zorder=2)
        
    ### Set extent of map
    axs[position].set_extent(coords, 
                             crs=ccrs.PlateCarree())
    
    ### Plot country borders for all regions except global
    if region != 'global':
        ### Add state borders
        axs[position].add_feature(cartopy.feature.BORDERS)
    
    ### Plot two colorsbars
    if position in (0,3):
        ### Set location, size and label of colorbar
        if var == 'natural':
            cax = plt.axes([0.15, 0.59, 0.7, 0.035])
            label = '$\Delta$ Cover [%]'
        if var == 'cVeg':
            cax = plt.axes([0.05, 0.08, 0.9, 0.035])
            label = '$\Delta$ Carbon pool [gC m$^{-2}$]'
        if var == 'tas':
            cax = plt.axes([0.15, 0.59, 0.7, 0.035])
            label = '$\Delta$ T [K]'
        if var == 'pr':
            cax = plt.axes([0.15, 0.08, 0.7, 0.035])
            label = '$\Delta$ PPT [mm yr$^{-1}$]'
        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend='both', 
                    )   
        
        ### Reduce fontsize
        cbar.ax.tick_params(axis='x', labelsize=8)  
        cbar.set_label(label=label,fontsize=10)  
                         
    ### Reduce fontsize of coordinate labels
    axs[position].tick_params(axis='both', labelsize=8)     

if region == 'global':
    figsize=(15,6)
    plot_params = {'top':0.99, 'left':0.025,
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.08, 'hspace':0.25}
if region == 'ACTO':
    figsize=(10,7)
    plot_params = {'top':0.95, 'left':0.05,
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.1, 'hspace':0.45}
if region == 'ASEAN':
    figsize=(10,7)
    plot_params = {'top':0.95, 'left':0.035,
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.07, 'hspace':0.45}
if region == 'EAC':
    figsize=(12,7)
    plot_params = {'top':0.95, 'left':0.05,
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.05, 'hspace':0.45} 
if region == 'EU27':
    figsize=(9,7)
    plot_params = {'top':0.95, 'left':0.025,
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.0, 'hspace':0.45}
if region == 'NAMERICA':
    figsize=(11,7)
    plot_params = {'top':0.95, 'left':0.025, 
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.0, 'hspace':0.45}
if region == 'OCEANIA':
    figsize=(10,7)
    plot_params = {'top':0.95, 'left':0.05,
                   'right':0.975, 'bottom':0.15,
                   'wspace':0.1, 'hspace':0.45}    

fig, axs = plt.subplots(nrows=2,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)
axs=axs.flatten()

def plot_LCF_cPool(exp,ref,first_year,last_year):
    global fig
    global axs
    global plot_params
    
    var_short_names = ['natural','pasture','cropland',
                       'cVeg','cBECCS','cBiochar']
    var_titles = ['Natural vegetation','Pasture','Cropland',
                  'Carbon stored in vegetation',
                  'Carbon stored in BECCS',
                  'Carbon stored in biochar']
    positions=[0,1,2,3,4,5]
    title_index=['a)','b)','c)','d)','e)','f)']

    ### Loop through plot command, and adjust subplot titles
    for vars, vart, p, ti in zip(var_short_names, var_titles, positions, title_index):
        make_map(vars,first_year,last_year,exp,ref,p,region)
        axs[p].set_title(vart,fontsize=10)
        axs[p].set_title(ti, loc='left',fontsize=10)

    ### Adjust ticklabels on axes
    for p in (0,3):
        ### Show ticklabels left
        axs[p].yaxis.set_visible(True)

    for p in (3,4,5):
        ### Show ticklabels bottom
        axs[p].xaxis.set_visible(True)

    plt.subplots_adjust(**plot_params)
    plt.savefig('figures/LCF_cPool_'+exp+'_'+ref+'_'+first_year+'-'+last_year+'_sens_'+region+'.png',dpi=400)

def plot_clim(first_year,last_year):
    global fig
    global axs
    global plot_params
    
    exp_short_names = ['a6xv','a6xx']
    exp_titles = ['Moderate ambition','High ambition']
    positions=[0,1]
    title_index=['a)','b)']

    ### Loop through plot command, and adjust subplot titles
    for exps, expt,p,ti in zip(exp_short_names, exp_titles,positions,title_index):
        make_map('tas',first_year,last_year,exps,'a6zt',p,region)
        axs[p].set_title(expt,fontsize=10)
        axs[p].set_title(ti, loc='left',fontsize=10)

    positions=[3,4]
    title_index=['d)','e)']

    ### Loop through plot command, and adjust subplot titles
    for exps, expt,p,ti in zip(exp_short_names, exp_titles,positions,title_index):
        make_map('pr',first_year,last_year,exps,'a6zt',p,region)
        axs[p].set_title(expt,fontsize=10)
        axs[p].set_title(ti, loc='left',fontsize=10)

    vars=['tas','pr']
    positions=[2,5]
    title_index=['c)','f)']

    ### Loop through plot command, and adjust subplot titles
    for var,p,ti in zip(vars,positions,title_index):
        make_map(var,first_year,last_year,'a6xx','a6xv',p,region)
        axs[p].set_title('High vs moderate ambition',fontsize=10)
        axs[p].set_title(ti, loc='left',fontsize=10)
            
    ### Adjust ticklabels on axes
    for p in (0,3):
        ### Show ticklabels left
        axs[p].yaxis.set_visible(True)

    for p in (3,4,5):
        ### Show ticklabels bottom
        axs[p].xaxis.set_visible(True)

    plt.subplots_adjust(**plot_params)

    plt.savefig('figures/clim_'+first_year+'-'+last_year+'_sens_'+region+'.png',dpi=400)

if args.plot_type == 'clim':
    plot_clim(first_year,last_year)
else:
    plot_LCF_cPool(args.exp,args.ref,first_year,last_year)    