import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from  matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse

'''
Initialise argument parsing: 
exp for LMT scenario (a6jn, a6xv,a6xx,a6xy)
first_year and last_year for fut. projection timeslice (2045-2059 or 2085-2099)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)
parser.add_argument('--climate_stats', type=str, required=True)
parser.add_argument('--paper_part', type=str, required=True)

args = parser.parse_args()

### Assign variables
first_year=args.first_year
last_year=args.last_year
climate_stats=args.climate_stats
paper_part=args.paper_part

### Set pathway where input files are located
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc').sel(lat=slice(-60, 90))

### Get gridarea
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc').sel(lat=slice(-60, 90))
ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']
da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf']))

### Map future projection realisations to 
### their corresponding historical experiment
hist_projection_mapping = {
    'r1i1p1f1': 'a3bh',
    'r2i1p1f1': 'a3o0',
    'r3i1p1f1': 'a3nm'
}
hist_realisation_mapping = {
    'a3bh': 'r1i1p1f1',
    'a3o0': 'r6i1p1f1',
    'a3nm': 'r7i1p1f1'
}
       
def time_aggr(ds,var,time_scale):
    ds['lat'], ds['lon'] = ds_mask['lat'], ds_mask['lon']
    ### Set up resampling
    if time_scale == 'annual':
        resample_freq = 'YS'
    else:
        resample_freq = 'QS-DEC'
        
    ### Temporal aggregation
    if var == 'pr':
        ds_time_scale = ds.resample(time=resample_freq).sum()
    else:
        ### Get days per month
        da_DPM = ds.time.dt.days_in_month
            
        ### Multiply variable with days per month
        ds[var] = ds[var] * da_DPM  
        
        ds_time_scale = ds.resample(time=resample_freq).sum() / \
                        da_DPM.resample(time=resample_freq).sum()

    ### If seasonal: Select season
    if time_scale != 'annual':
            ds_time_scale = ds_time_scale.sel(time=ds_time_scale.time.dt.season == time_scale)

    return(ds_time_scale)
        
def get_data(var,time_scale,realisation,first_year,last_year,exp_id):
    global pathwayIN
    global ds_mask

    if exp_id == 'a7en' and realisation == 'r2i1p1f1':
        realisation = 'r4i1p1f1'
            
    if var in ('tx90p','dry_days'):
        ID = 'day'
        suffix = '1850-2100.nc'
    else:
        ID = 'Amon'
        suffix = '185001-210012.nc'
    
    ds_raw = xr.open_dataset(pathwayIN+exp_id+'_'+realisation+'/climate/'+
                             var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+realisation+
                             '_gr_'+suffix).sel(time=slice(first_year,last_year),
                                                lat=slice(-60, 90))
                                        
    ds = time_aggr(ds_raw,var,time_scale)
    
    ### Set ocean pixels to nan
    da = ds[var].where(~np.isnan(ds_mask['sftlf']))

    if 'height' in da.coords:
        da = da.drop_vars('height', errors='ignore')
            
    return(da.mean(dim='time'))

def get_zonal_average(da):
    ### Get zonal sum - weighted by gridarea and then sum across longitudes
    da_weighted = da * da_gridarea
    da_zonal = da_weighted.sum(dim='lon')/da_gridarea.sum(dim='lon')
    return(da_zonal)

def get_diff(var,time_scale,exp_id,exp_ref,realisation,first_year,last_year):
    ### Get data for experiment as map and zonal sum
    if exp_ref == exp_id:
        first_year_ref, last_year_ref = '1981', '2010'
    else:
        first_year_ref, last_year_ref = first_year, last_year
        
    da_ref = get_data(var,time_scale,realisation,first_year_ref,last_year_ref,exp_ref)
    da_ref_zonal = get_zonal_average(da_ref)
    
    da_exp = get_data(var,time_scale,realisation,first_year,last_year,exp_id)
    da_exp_zonal = get_zonal_average(da_exp)
        
    ### Get difference maps
    da_diff = da_exp - da_ref
        
    ### Calculate zonal sums
    da_zonal_diff = da_exp_zonal - da_ref_zonal
    
    ### Return map and zonal sum
    return(da_diff, da_zonal_diff)

def get_ensmean(var,time_scale,exp_id,exp_ref,first_year,last_year):
    ### List of realisations
    realisations = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']

    ### Get data
    das_diff, das_zonal_diff = zip(*[
        get_diff(var, time_scale, exp_id, exp_ref, realisation, first_year, last_year) for realisation in realisations
    ])
    
    ### Merge along dimension
    da_diff = xr.concat(das_diff, dim='realization') 
    da_zonal_diff = xr.concat(das_zonal_diff, dim='realization')

    ### Return DataArrays
    return(da_diff, da_zonal_diff)

### Set up plot for maps
def make_map(var,time_scale,first_year,last_year,exp_id,ref,position):  
    ### Get data
    da, da_zonal = get_ensmean(var,time_scale,exp_id,ref,first_year,last_year)
    
    ### Calculate ensmean
    da_ensmean = da.mean(dim='realization')   
    da_zonal_ensmean = da_zonal.mean(dim='realization')    
    da_zonal_ensstd = da_zonal.std(dim='realization')    
    
    ### Get latitude and longitude coordinates
    lat, lon = da_ensmean.lat, da_ensmean.lon
            
    ### Set levels for colorbar
    if var == 'tas':
        cmap = 'RdBu_r'
        full_levels = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,-0.1, 0.1,0.5,1,1.5,2,2.5,3,3.5,4]
        display_levels = [-1,-0.5,-0.1, 0.1,0.5,1,1.5,2,2.5,3,3.5,4]
    elif var == 'pr':
        cmap = 'BrBG'
        full_levels = [-400,-200,-100,-40,-20,-10,10,20,40,100,200,400]
        display_levels = full_levels
    elif var == 'tx90p':
        cmap = 'RdBu_r'
        full_levels = [-180,-160,-140,-120,-100,-80,-60,-40,-20,-5,5,20,40,60,80,100,120,140,160,180]
        display_levels = [-40,-20,-5,5,20,40,60,80,100,120,140,160,180]
    elif var == 'dry_days':
        cmap = 'BrBG_r'
        full_levels = [-30,-25,-20,-15,-10,-5,-1,1,5,10,15,20,25,30]
        display_levels = full_levels
                
    ### Set colorbar: assign new colors to arrow tips so add two extra colors
    pal = sns.color_palette(cmap, len(full_levels)+2)
    cols = pal.as_hex()

    ### Set low values to grey
    cols[int(len(full_levels)/2)] = '#e4e4e4'

    ### cmap based on actual number of levels (exclude first and last cmap val)
    if var in ('tas','tx90p'):
        cols[8] = '#bfdbeb'
        cols[7] = '#7eb7d7'
        cmap = ListedColormap(cols[7:-1])

        ### Set arrow color to first and last cmap val
        cmap.set_under('#5594b8')
        cmap.set_over(cols[-1])
    else:
        cmap = ListedColormap(cols[1:-1])

        ### Set arrow color to first and last cmap val
        cmap.set_under(cols[0])
        cmap.set_over(cols[-1])   
    
    ### Add cyclic point
    da_ensmean_cyclic, lon = add_cyclic_point(da_ensmean, coord=lon)
    
    ### Set projection
    projection = ccrs.PlateCarree()   

    ### Boundary norm: set ncolors to ACTUAL number of levels (excl. first and last cmap val)      
    bounds = display_levels
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da_ensmean, cmap=cmap, norm=norm,zorder=1)
            
    ### Get map with robust changes
    if var in ('tx90p','dry_days'):
        ID = 'day'
    else:
        ID = 'Amon'
    
    da_robust = xr.open_dataset(pathwayIN+'robustness_maps/'+first_year+'-'+last_year+
                                '/climate/'+var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+ref+'_'+
                                exp_id+'_gr.nc').sel(lat=slice(-60, 90))['pvals'][0]
    
    ### Mask ocean
    da_robust = da_robust.where(~np.isnan(ds_mask['sftlf']))
    da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf']))
                           
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
        'pos': (da_ensmean >= 0) & (da_robust == 1),
        'neg': (da_ensmean <= 0) & (da_robust == 1)
    }

    percs = {
        key: 100 * ((da_gridarea * mask).sum(dim=['lat', 'lon']) / 
                    tmp.sum(dim=['lat', 'lon']))
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
                          bbox_to_anchor=(0.15, 0, 1, 1),
                          bbox_transform=axs[position].transAxes)
       
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
    
    ### Set x-label
    if var == 'tas':
        label = '$\Delta$T [K]'
    elif var == 'pr':
        label = '$\Delta$pr [mm]'
    elif var == 'tx90p':
        label = '$\Delta$TX90p [days]'
    elif var == 'dry_days':
        label = '$\Delta$Dry days'
    
    if position in (4,5):
        inset_ax.set_xlabel(label,fontsize=7,labelpad=0.8)
    
    ### Remove spines
    inset_ax.spines['left'].set_position(('outward', 1))
    inset_ax.spines['left'].set_linewidth(0.6)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    
    ### Fix x lim range
    if position in (2,4):
        if ref != exp_id:
            if var == 'tas':
                inset_ax.set_xlim(-1.1,0.1)
            else:
                inset_ax.set_xlim(-27,3)
            
    elif position in (3,5):
        if ref != exp_id:
            if var == 'pr':
                inset_ax.set_xlim(-130,40)
            else:
                inset_ax.set_xlim(-10,7)
        
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
        
    ## Plot two colorsbars
    if position == 4:
        ### left, bottom, width, height
        cax = plt.axes([0.05, 0.06, 0.45, 0.025])                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=display_levels, 
                    orientation='horizontal',
                    extend='both', 
                    )   
        
        ### Reduce fontsize
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label=label,fontsize=8)  
    
    elif position == 5:
        cax = plt.axes([0.51, 0.06, 0.45, 0.025])              
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=display_levels, 
                    orientation='horizontal',
                    extend='both', 
                    )
        ### Reduce fontsize
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label,fontsize=8)  
            
    ### Reduce fontsize of coordinate labels
    axs[position].tick_params(axis='both', labelsize=8)     

figsize=(8,6.3)
plot_params = {'top':0.96, 'left':0.05,
               'right':0.93, 'bottom':0.1,
               'wspace':0.2, 'hspace':0}

fig, axs = plt.subplots(nrows=3,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)
axs=axs.flatten()

def call_plot(climate_stats,paper_part):
    time_scale='annual'
    
    if paper_part == 'main':
        exp_pert_row_1, exp_ref_row_1 = 'a7cy', 'a7cy'
        exp_pert_row_2, exp_ref_row_2 = 'a7en', 'a7cy'
        exp_pert_row_3, exp_ref_row_3 = 'a7eo', 'a7en'

    elif paper_part == 'supplement':
        exp_pert_row_1, exp_ref_row_1 = 'a7en', 'a7en'
        exp_pert_row_2, exp_ref_row_2 = 'a7eo', 'a7eo'
        exp_pert_row_3, exp_ref_row_3 = 'a7eo', 'a7cy'
        
    if climate_stats == 'mean_climate':
        var_left = 'tas'
        var_right = 'pr'
        title_left = 'Temperature'
        title_right = 'Precipitation'
    elif climate_stats == 'climate_extremes':
        var_left = 'tx90p'
        var_right = 'dry_days'
        title_left = 'TX90p'
        title_right = 'Dry days'
        
    make_map(var_left,time_scale,first_year,last_year,exp_pert_row_1, exp_ref_row_1,0)
    make_map(var_left,time_scale,first_year,last_year,exp_pert_row_2, exp_ref_row_2,2)
    make_map(var_left,time_scale,first_year,last_year,exp_pert_row_3, exp_ref_row_3,4)
    make_map(var_right,time_scale,first_year,last_year,exp_pert_row_1, exp_ref_row_1,1)
    make_map(var_right,time_scale,first_year,last_year,exp_pert_row_2, exp_ref_row_2,3)
    make_map(var_right,time_scale,first_year,last_year,exp_pert_row_3, exp_ref_row_3,5)

    if paper_part == 'main':
        axs[0].set_title(title_left+'\nReference (2071-2100 vs. 1981-2010)',
                        fontsize=8,linespacing=1.75)
        axs[1].set_title(title_right+'\nReference (2071-2100 vs. 1981-2010)',
                        fontsize=8,linespacing=1.75)
        axs[2].set_title('Moderate CDR Ambition vs. Reference (2071-2100)',fontsize=8)
        axs[3].set_title('Moderate CDR Ambition vs. Reference (2071-2100)',fontsize=8)
        axs[4].set_title('High vs. Moderate CDR Ambition (2071-2100)',fontsize=8)
        axs[5].set_title('High vs. Moderate CDR Ambition (2071-2100)',fontsize=8)
    else:
        axs[0].set_title(title_left+'\nModerate CDR Ambition (2071-2100 vs. 1981-2010)',
                        fontsize=8,linespacing=1.75)
        axs[1].set_title(title_right+'\nModerate CDR Ambition (2071-2100 vs. 1981-2010)',
                        fontsize=8,linespacing=1.75)
        axs[2].set_title('High CDR Ambition (2071-2100 vs. 1981-2010)',fontsize=8)
        axs[3].set_title('High CDR Ambition (2071-2100 vs. 1981-2010)',fontsize=8)
        axs[4].set_title('High CDR Ambition vs. Reference (2071-2100)',fontsize=8)
        axs[5].set_title('High CDR Ambition vs. Reference (2071-2100)',fontsize=8)
        
    positions=[0,1,
            2,3,
            4,5]
    title_index=['a)','b)',
                'c)','d)',
                'e)','f)']

    ### Loop through plot command, and adjust subplot titles
    for p, ti in zip(positions, title_index):
        axs[p].set_title(ti, loc='left',fontsize=8)

    ### Adjust ticklabels on axes
    for p in (0,2,4):
        ### Show ticklabels left
        axs[p].yaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelsize=8)

    for p in (4,5):
        ### Show ticklabels bottom
        axs[p].xaxis.set_visible(True)
        axs[p].tick_params(axis='x', labelsize=8)
        
    for p in (1,3,5):
        axs[p].yaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelleft=False)
    
    plt.subplots_adjust(**plot_params)
    plt.savefig('figures/climate/maps_'+climate_stats+'_'+time_scale+
                '_'+first_year+'-'+last_year+'_'+paper_part+'.png',dpi=400)
       
call_plot(climate_stats,paper_part)        
