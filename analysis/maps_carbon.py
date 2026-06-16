import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from  matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

import argparse

'''
Initialise argument parsing: 
exp for LMT scenario (a6jn, a6xv,a6xx,a6xy)
first_year and last_year for fut. projection timeslice (2045-2059 or 2085-2099)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--pool_type', type=str, required=True)
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)

args = parser.parse_args()

### Assign variables
pool_type=args.pool_type
first_year=args.first_year
last_year=args.last_year

### Set pathway where input files are located
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

# ### Get gridarea
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc')
ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']
da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf']))

### Define cmor IDs
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
    if np.isclose(x, 0):
        return '0'
    else:
        return f'{x:g}'
      
def get_data(var,exp,realisation,first_year,last_year): 
    ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
    if exp == 'a7en' and realisation == 'r2i1p1f1':
        realisation = 'r4i1p1f1'
        
    ### Get CMIP ID
    ID = var_to_ID_dict[var]
    directory = var_to_directory_dict[var]
    
    ### Climate files are merged to 1850-2100, carbon files are split into historical and projection
    if exp  == 'a766':
        first_year, last_year = '1981', '2010'
        CMIP_scen, suffix, realisation = 'historical', '1850-2014.nc', 'r1i1p1f1'
    else:
        if directory == 'carbon':
            CMIP_scen, suffix = 'ssp245', '2015-2100.nc'
        else:
            CMIP_scen, suffix = 'ssp245', '185001-210012.nc'
    
    ### cBECCS, cBiochar and fco2fossub are 0 in historical - no need to read in
    if not (var in ('cBECCS', 'cBiochar', 'fco2fossub') and exp == 'a766'):
        ### Create file name
        fname=(pathwayIN+exp+'_'+realisation+'/'+directory+'/'+
               var+'/'+var+'_'+ID+'_EC-Earth3-CC_'+CMIP_scen+
               '_'+realisation+'_gr_'+suffix)
        print(fname)
        ### Open dataset
        da = xr.open_dataset(fname)[var]
        
        ### Exclude Antarctica and make sure coordinates match 
        da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon'] 
                
        ### Cumulative sum for fco2fossub
        if var == 'fco2fossub':
            da_final = da.cumsum(dim='time')
        else:
            da_final = da
        
        ### Mask ocean
        da_final_mask = da_final.sel(time=slice(first_year,last_year)).mean(dim='time').where(ds_mask['sftlf'] != 0, np.nan)   
        
        ### Return dataset
        return(da_final_mask)

    ### 0 during historical - generate fake data based on cLand
    elif var in ('cBECCS', 'cBiochar', 'fco2fossub') and exp == 'a766':
        fname=(pathwayIN+exp+'_'+realisation+'/'+directory+'/cLand/cLand_Emon_EC-Earth3-CC_'+
               CMIP_scen+'_'+realisation+'_gr_'+suffix)
        print(fname)
        
        ### Open dataset
        da = xr.open_dataset(fname)['cLand']

        ### Exclude Antarctica and make sure coordinates match 
        da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon'] 
                
        ### Cumulative sum for fco2fossub
        da_final = da[0] * 0
            
        ### Return dataset
        return(da_final)
    
def get_zonal_sum(da):
    ### Get zonal sum - weighted by gridarea and then sum across longitudes
    da_weighted = da * da_gridarea
    da_zonal = da_weighted.sum(dim='lon')
    return(da_zonal)

def get_diff(var,exp_ref,exp_pert,realisation,first_year,last_year):
    ### Get data for experiment as map and zonal sum
    ### Get difference within the same simulation or compare two experiments
    if exp_pert == exp_ref:
        first_year_ref, last_year_ref = '1981', '2010'
        if var == 'cLand':
            exp_ref = 'a766'
    else:
        first_year_ref, last_year_ref = first_year, last_year
    
    ### Read in data
    da_ref = get_data(var,exp_ref,realisation,first_year_ref,last_year_ref)
    da_exp = get_data(var,exp_pert,realisation,first_year,last_year)
    
    ### Align coordinates
    da_ref['lat'],da_ref['lon'] = da_exp['lat'], da_exp['lon']    
     
    ### Get zonal sums
    da_ref_zonal = get_zonal_sum(da_ref)
    da_exp_zonal = get_zonal_sum(da_exp)
    
    ### Get difference maps, convert kgC m-2 to gC m-2
    da_diff = (da_exp - da_ref)*1000
    
    ### Calculate zonal sums and convert to PgC
    da_zonal_diff = (da_exp_zonal - da_ref_zonal) / 1e+12       

    ### Return map and zonal sum
    return(da_diff, da_zonal_diff)

def get_ensmean(var,exp_ref,exp_pert,first_year,last_year):
    ### List of realisations
    realisations = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']

    ### Get data
    das_diff, das_zonal_diff = zip(*[
        get_diff(var, exp_ref, exp_pert, realisation, first_year, last_year) for realisation in realisations
    ])
    
    ### Merge along dimension, adjust unit
    da_diff = xr.concat(das_diff, dim='realization')
    da_zonal_diff = xr.concat(das_zonal_diff, dim='realization')

    ### Return DataArrays
    return(da_diff, da_zonal_diff)

### Set up plot for maps
def make_map(var,first_year,last_year,exp_ref,exp_pert,position):  
    ### ### Get CMIP ID
    ID = var_to_ID_dict[var]   
        
    ### Get data for experiment    
    da, da_zonal = get_ensmean(var,exp_ref,exp_pert,first_year,last_year)
    
    ### Calculate ensmean
    da_ensmean = da.mean(dim='realization')   
    da_zonal_ensmean = da_zonal.mean(dim='realization')    
     
    ### Calculate ensemble standard deviation
    da_zonal_ensstd = da_zonal.std(dim='realization')
        
    ### Mask ocean
    da_ensmean = da_ensmean.where(~np.isnan(ds_mask['sftlf']))
        
    ### Get latitude and longitude coordinates
    lat, lon = da_ensmean.lat, da_ensmean.lon

    ### Set levels for colorbar
    cmap = 'BrBG'
    if var == 'cVeg':
        if position == 0:
            levels = [-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,-50,
                      50,500,1000,1500,2000,2500,3000,3500,4000]
        else:
            levels = [-2000,-1750,-1500,-1250,-1000,-750,-500,-250,-50,
                      50,250,500,750,1000,1250,1500,1750,2000]   
    else:
            levels = [-800,-700,-600,-500,-400,-300,-200,-100,-50,
                      50,100,200,300,400,500,600,700,800]      

    ### Set colorbar: assign new colors to arrow tips so add two extra colors
    pal = sns.color_palette(cmap, len(levels)+2)
    cols = pal.as_hex()

    ### Set low values to grey
    cols[int(len(levels)/2)] = '#e4e4e4'

    ### cmap based on actual number of levels (exclude first and last cmap val)
    cmap = ListedColormap(cols[1:-1])

    ### Set arrow color to first and last cmap val
    cmap.set_under(cols[0])
    cmap.set_over(cols[-1]) 

    ### Add cyclic point to da
    da_ensmean_cyclic, lon = add_cyclic_point(da_ensmean, coord=lon)
    
    ### Set projection
    projection = ccrs.PlateCarree()   

    ### Boundary norm: set ncolors to ACTUAL number of levels (excl. first and last cmap val)      
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da_ensmean_cyclic, cmap=cmap, norm=norm,zorder=1)

    ### Read in file with significant change mask
    da_robust = xr.open_dataset(pathwayIN+'robustness_maps/'+first_year+'-'+last_year+
                                '/carbon/'+var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+exp_ref+'_'+
                                exp_pert+'_gr.nc')['pvals'][0]
    
    ### Mask ocean points
    da_robust['lat'],da_robust['lon'] = ds_mask['lat'], ds_mask['lon']    
    da_robust = da_robust.where(~np.isnan(ds_mask['sftlf']))
    
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
    lon_coords = da_robust.lon[lon_robust]
    lat_coords = da_robust.lat[lat_robust]

    ### Add scatter for stippling
    axs[position].scatter(lon_coords,
                          lat_coords,
                          marker=',',
                          s=marker_size,
                          color='k',
                          transform=ccrs.PlateCarree())

    ### Get percentage of significant landarea
    tmp = da_gridarea * da_robust
    perc_sign = 100 * (tmp.sum(dim=['lat', 'lon']) / \
                       da_gridarea.sum(dim=['lat', 'lon']))

    ### Get percentage of landarea with sign. positive and negative values
    masks = {
        'pos': (da_ensmean.sel(lat=slice(-60,90)) >= 0) & (da_robust.sel(lat=slice(-60,90)) == 1),
        'neg': (da_ensmean.sel(lat=slice(-60,90)) <= 0) & (da_robust.sel(lat=slice(-60,90)) == 1)
    }

    percs = {
        key: 100 * ((da_gridarea.sel(lat=slice(-60,90)) * mask).sum(dim=['lat', 'lon']) / 
                    tmp.sel(lat=slice(-60,90)).sum(dim=['lat', 'lon']))
        for key, mask in masks.items()
    }

    ### Round results
    perc_sign = perc_sign.round(2)
    perc_sign_pos = percs['pos'].round(2)
    perc_sign_neg = percs['neg'].round(2)
    
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
        
    ### Plot zonal sum to the right
    ### Set up axis
    inset_ax = inset_axes(axs[position],
                        width='10%',
                        height='100%',
                        loc='center right',
                        bbox_to_anchor=(0.125, 0, 1, 1),
                        bbox_transform=axs[position].transAxes)
        
    ### Plot zonal sum       
    inset_ax.plot(da_zonal_ensmean.values,
                  da_zonal_ensmean.lat,color='#00678a')

    ### Shading of uncertainty
    inset_ax.fill_betweenx(da_zonal_ensmean.lat,
                           da_zonal_ensmean.values-da_zonal_ensstd.values,
                           da_zonal_ensmean.values+da_zonal_ensstd.values,
                           color='#00678a',alpha=0.4)
    
    ### Plot hline at 0
    inset_ax.axvline(0,linewidth=0.5,color='tab:grey')
    
    ### Align y-axis with latitudes of map  
    inset_ax.set_ylim(axs[position].get_ylim())
    
    ### Remove yticks
    inset_ax.set_yticks([])
    
    ### Set fontsize for x ticks
    inset_ax.tick_params(axis='x', labelsize=8)
    inset_ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    
    ### Remove spines
    inset_ax.spines['left'].set_position(('outward', 1))
    inset_ax.spines['left'].set_linewidth(0.6)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
        
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
     
    ### Plot two colorsbars
    if position == 0:
        cax = plt.axes([0.03, 0.65, 0.26, 0.025]) 
        label = 'Carbon [gC m$^2$]'
                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend='both', 
                    )   
        
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label='$\Delta$ '+label,fontsize=10,labelpad=15)
        cbar.ax.xaxis.set_label_position('top')

        if var in ('cBECCS','cBiochar','fco2fossub'):
            ticks_above = {-800,-600,-400,-200,-50,50,200,400,600,800}
            ticks_below = {-700,-500,-300,-100,100,300,500,700}
        else:
            ticks_above = {-3500,-2500,-1500,-500,500,1500,2500,3500}
            ticks_below = {-4000,-3000,-2000,-1000,-50,50,1000,2000,3000,4000}
        
        for label in cbar.ax.get_xticklabels():
            # label_value = float(label.get_text())
            label_value = float(label.get_text().replace('−', '-'))  # Replace Unicode minus
            if label_value in ticks_above:
                label.set_va('top') 
                label.set_position((label.get_position()[0], 2.2)) # Move label above
            elif label_value in ticks_below:
                label.set_va('bottom')
                label.set_position((label.get_position()[0], -0.5))

    elif position == 2:
        ### left, bottom, width, height
        cax = plt.axes([0.5, 0.65, 0.32, 0.025])
        label = 'Carbon [gC m$^2$]'
                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend='both', 
                    )   
        
        cbar.set_label(label='$\Delta$ '+label,fontsize=10,labelpad=15)
        cbar.ax.tick_params(axis='x', labelsize=8)
        cbar.ax.xaxis.set_label_position('top')

        if var in ('cBECCS','cBiochar','fco2fossub'):
            ticks_above = {-800,-600,-400,-200,-50,50,200,400,600,800}
            ticks_below = {-700,-500,-300,-100,100,300,500,700} 
        else:       
            ticks_above = {-1750,-1250,-750,-250,250,750,1250,1750}
            ticks_below = {-2000,-1500,-1000,-500,-50,50,500,1000,1500,2000}

        for label in cbar.ax.get_xticklabels():
            # label_value = float(label.get_text())
            label_value = float(label.get_text().replace('−', '-'))  # Replace Unicode minus
            if label_value in ticks_above:
                label.set_va('top') 
                label.set_position((label.get_position()[0], 2.2)) # Move label above
            elif label_value in ticks_below:
                label.set_va('bottom')
                label.set_position((label.get_position()[0], -0.5))  
        
    elif position == 4:
        ### left, bottom, width, height
        cax = plt.axes([0.25, 0.05, 0.5, 0.025])
        label = 'Carbon [gC m$^2$]'
                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend='both', 
                    )   
        
        ### Reduce fontsize
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label='$\Delta$ '+label,fontsize=10)
        cbar.ax.xaxis.set_label_position('top') 

### Set up figure    
figsize=(14,8)
plot_params = {'top':0.96, 'left':0.03,
               'right':0.96, 'bottom':0.03,
               'wspace':0.15, 'hspace':0.5}
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)
axs=axs.flatten()

### Decide which carbon pools to plot
if pool_type == 'natural':
    cpool_1, cpool_2, cpool_3 = 'cVeg','cSoil','cLitter'
else:
    cpool_1, cpool_2, cpool_3 = 'cBECCS','cBiochar','fco2fossub'

### Plot carbon pools
make_map(cpool_1,first_year,last_year,'a766','a7cy',0)
make_map(cpool_1,first_year,last_year,'a7cy','a7en',1)
make_map(cpool_1,first_year,last_year,'a7en','a7eo',2)

make_map(cpool_2,first_year,last_year,'a766','a7cy',3)
make_map(cpool_2,first_year,last_year,'a7cy','a7en',4)
make_map(cpool_2,first_year,last_year,'a7en','a7eo',5)

make_map(cpool_3,first_year,last_year,'a766','a7cy',6)
make_map(cpool_3,first_year,last_year,'a7cy','a7en',7)
make_map(cpool_3,first_year,last_year,'a7en','a7eo',8)

### Plot subplot titles
if pool_type == 'natural':
        ## Plot subplot titles
        axs[0].set_title('Reference (2071-2100 vs. 1981-2010)\nCarbon stored in vegetation',
                         fontsize=9,linespacing=1.75)
        axs[3].set_title('Carbon stored in soil',fontsize=10)
        axs[6].set_title('Carbon stored in litter',fontsize=10)
        
        axs[1].set_title('Moderate CDR Ambition vs. Reference (2071-2100)\nCarbon stored in vegetation',
                         fontsize=9,linespacing=1.75)
        axs[4].set_title('Carbon stored in soil',fontsize=10)
        axs[7].set_title('Carbon stored in litter',fontsize=10)

        axs[2].set_title('High vs. Moderate CDR Ambition (2071-2100)\nCarbon stored in vegetation',
                         fontsize=9,linespacing=1.75)                        
        axs[5].set_title('Carbon stored in soil',fontsize=10)
        axs[8].set_title('Carbon stored in litter',fontsize=10)
else:
        ## Plot subplot titles
        axs[0].set_title('Reference (2071-2100 vs. 1981-2010)\nCarbon stored in BECCS',
                         fontsize=9,linespacing=1.75)
        axs[3].set_title('Carbon stored in biochar',fontsize=10)
        axs[6].set_title('Fossil fuel substitution',fontsize=10)
        
        axs[1].set_title('Moderate CDR Ambition vs. Reference (2071-2100)\nCarbon stored in BECCS',
                         fontsize=9,linespacing=1.75)
        axs[4].set_title('Carbon stored in biochar',fontsize=10)
        axs[7].set_title('Fossil fuel substitution',fontsize=10)

        axs[2].set_title('High vs. Moderate CDR Ambition (2071-2100)\nBECCS',
                         fontsize=9,linespacing=1.75)                        
        axs[5].set_title('Carbon stored in biochar',fontsize=10)
        axs[8].set_title('Fossil fuel substitution',fontsize=10)
        

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

for p in (6,7,8):
    ### Show ticklabels bottom
    axs[p].xaxis.set_visible(True)
    axs[p].tick_params(axis='x', labelsize=8)

for p in (0,1,2,3,4,5):
    axs[p].xaxis.set_visible(True)
    axs[p].tick_params(axis='x', labelbottom=False)

for p in (1,2,4,5,7,8):
    axs[p].yaxis.set_visible(True)
    axs[p].tick_params(axis='y', labelleft=False)
           
plt.subplots_adjust(**plot_params)
# Get current position of a bottom subplot: [left, bottom, width, height]
pos_6 = axs[6].get_position()
pos_7 = axs[7].get_position()  
pos_8 = axs[8].get_position()

# Shift up: reduce bottom by 0.08 (or tune), reduce height slightly to fit
new_pos_6 = [pos_6.x0, pos_6.y0 + 0.08, pos_6.width, pos_6.height]
new_pos_7 = [pos_7.x0, pos_7.y0 + 0.08, pos_7.width, pos_7.height]
new_pos_8 = [pos_8.x0, pos_8.y0 + 0.08, pos_8.width, pos_8.height]

# Apply to all three bottom panels
axs[6].set_position(new_pos_6)
axs[7].set_position(new_pos_7)
axs[8].set_position(new_pos_8)

plt.savefig('figures/carbon/maps_carbon_'+pool_type+'_'+
            first_year+'-'+last_year+'.png',dpi=400)
