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

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc').sel(lat=slice(-60, 90))

# ### Get gridarea
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc').sel(lat=slice(-60, 90))
ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']
da_gridarea = ds_gridarea['cell_area'].where(ds_mask['sftlf'] != 0, np.nan)

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

def custom_formatter(x,pos):
    if x == 0:
        return('0')
    else:
        return(f'{x:.1f}')

def time_aggr(da,var):
    da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon']
    ### Set up resampling
    resample_freq = 'YS'

    ### Temporal aggregation
    if var == 'pr':
        da_annual = da.resample(time=resample_freq).sum()
    else:
        ### Get days per month
        da_DPM = da.time.dt.days_in_month
            
        ### Multiply variable with days per month
        da_weighted = da * da_DPM  
        
        ### Get annual values
        da_annual = da_weighted.resample(time=resample_freq).sum() / \
                    da_DPM.resample(time=resample_freq).sum()

    return(da_annual)

def get_cLand(exp,realisation,first_year,last_year):
    ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
    if exp == 'a7en' and realisation == 'r2i1p1f1':
        realisation = 'r4i1p1f1'
    
    ### For historical, cLand = cLandAll, for projections it's the sum of cLand, BECCS and Biochar  
    if exp == 'a766':
        da_cLandAll = get_data('cLand',exp,realisation,first_year,last_year)
    else:
        #### Read in separate carbon pools
        da_cLand = get_data('cLand',exp,realisation,first_year,last_year)
        da_cBECCS = get_data('cBECCS',exp,realisation,first_year,last_year)
        da_cBiochar = get_data('cBiochar',exp,realisation,first_year,last_year)
        
        ### Sum to get all carbon stored on land
        da_cLandAll = da_cLand + da_cBECCS + da_cBiochar
    
    return(da_cLandAll)
   
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
        da = da.sel(lat=slice(-60, 90))
        da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon'] 
                
        ### Cumulative sum for fco2fossub
        if var == 'fco2fossub':
            da_cumsum = da.cumsum(dim='time')

        if var in ('albedo', 'hfls'):
            da_final = time_aggr(da,var)
        else:
            da_final = da
            
        ### Return dataset
        da_final.sel(time=slice(first_year,last_year)).mean(dim='time').to_dataset(name=var).to_netcdf(var+'_'+exp+'_'+realisation+'.nc')
        return(da.sel(time=slice(first_year,last_year)).mean(dim='time'))

def get_zonal_sum(da):
    ### Get zonal sum - weighted by gridarea and then sum across longitudes
    da_weighted = da * da_gridarea
    da_zonal = da_weighted.sum(dim='lon')
    return(da_zonal)

def get_zonal_average(da):
    ### Get zonal sum - weighted by gridarea and then sum across longitudes
    da_weighted = da * da_gridarea
    da_zonal = da_weighted.sum(dim='lon')/da_gridarea.sum(dim='lon')
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
    if var == 'cLand':
        da_ref = get_cLand(exp_ref,realisation,first_year_ref,last_year_ref)
        da_exp = get_cLand(exp_pert,realisation,first_year,last_year)
    else:
        da_ref = get_data(var,exp_ref,realisation,first_year_ref,last_year_ref)
        da_exp = get_data(var,exp_pert,realisation,first_year,last_year)
                
    ### Get zonal sums (cLand) / averages (albedo, hlfs)
    if var == 'cLand':
        da_ref_zonal = get_zonal_sum(da_ref)
        da_exp_zonal = get_zonal_sum(da_exp)
        
        ### Get difference maps, convert kgC m-2 to gC m-2
        da_diff = (da_exp - da_ref)*1000
        
        ### Calculate zonal sums
        da_zonal_diff = (da_exp_zonal - da_ref_zonal) / 1e+12 # convert to PgC       
    else:
        da_ref_zonal = get_zonal_average(da_ref)
        da_exp_zonal = get_zonal_average(da_exp)
        
        ### Show albedo differences in %
        if var == 'albedo':
            ### Get difference maps
            da_diff = ((da_exp - da_ref)/da_ref) * 100
            
            ### Calculate zonal sums
            da_zonal_diff = ((da_exp_zonal - da_ref_zonal)/da_ref_zonal) * 100
        else:
            ### Get difference maps
            da_diff = da_exp - da_ref
            
            ### Calculate zonal sums
            da_zonal_diff = da_exp_zonal - da_ref_zonal
                        
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
    directory = var_to_directory_dict[var]

    ### Get data for experiment    
    da, da_zonal = get_ensmean(var,exp_ref,exp_pert,first_year,last_year)
    
    ### Calculate ensmean
    da_ensmean = da.mean(dim='realization')   
    da_zonal_ensmean = da_zonal.mean(dim='realization')    
     
    ### Calculate ensemble standard deviation
    da_zonal_ensstd = da_zonal.std(dim='realization')
    
    da_ensmean.to_dataset(name=var).to_netcdf(var+'_ensmean.nc')
    da_zonal_ensmean.to_dataset(name=var).to_netcdf(var+'_zonal_ensmean.nc')
    da_zonal_ensstd.to_dataset(name=var).to_netcdf(var+'_zonal_ensstd.nc')
    
    ### Get map with robust changes
    if exp_ref == 'a766':
        ref = exp_pert
    else:
        ref = exp_ref
    
    ### Mask ocean
    da_ensmean = da_ensmean.where(ds_mask['sftlf'] != 0, np.nan)
        
    ### Get latitude and longitude coordinates
    lat, lon = da_ensmean.lat, da_ensmean.lon

    if var == 'cLand':
        cmap = 'BrBG'
        if position == 6:
            levels = [-4000,-3500,-3000,-2500,-2000,-1500,-1000,-500,-50,
                      50,500,1000,1500,2000,2500,3000,3500,4000]
        else:
            levels = [-2000,-1750,-1500,-1250,-1000,-750,-500,-250,-50,
                      50,250,500,750,1000,1250,1500,1750,2000]
                        
    elif var == 'albedo':
        cmap = 'RdBu'
        if position == 0:
            levels = [-30,-25,-20,-15,-10,-5,-1,
                      1,5,10,15,20,25,30]
        else:
            levels = [-15,-12.5,-10,-7.5,-5,-2.5,-1,
                      1,2.5,5,7.5,10,12.5,15]
    elif var == 'hfls':
        cmap = 'RdBu_r'
        if position == 3:
            levels = [-15,-12.5,-10,-7.5,-5,-2.5,-0.5,0.5,2.5,5,7.5,10,12.5,15]
        else:
            levels = [-7.5,-6.25,-5,-3.75,-2.5,-1.25,-0.5,0.5,1.25,2.5,3.75,5,6.25,7.5]

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
    
    ### Boundary norm: set ncolors to ACTUAL number of levels (excl. first and last cmap val)      
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    ### Get mask with robust points
    ### Read in file with significant change mask
    if directory == 'carbon':
        suffix = '_gr.nc'
    else:
        suffix = '_gr_annual.nc'
        
    da_robust = xr.open_dataset(pathwayIN+'robustness_maps/'+first_year+'-'+last_year+
                                '/'+directory+'/'+var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+exp_pert+'_'+
                                ref+suffix).sel(lat=slice(-60, 90))['pvals']
    
    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da_ensmean_cyclic, cmap=cmap, norm=norm,zorder=1)
    
    ### Mask ocean points    
    da_robust = da_robust.where(ds_mask['sftlf'] != 0, np.nan)
    
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
    
    ### Remove yticks
    # Define your desired ticks
    y_ticks = [-50, -25, 0, 25, 50, 75]

    # Set the y-ticks and labels
    inset_ax.set_yticks(y_ticks)
    
    inset_ax.set_yticklabels([])
    
    ### Set fontsize for x ticks
    inset_ax.tick_params(axis='x', labelsize=8)
    
    if exp_ref == 'a766' or (exp_ref == exp_pert):
        pass
    else:
        if var == 'albedo':
            inset_ax.set_xlim(-5.1,5.8)
            if position == 6:
                pass
                # inset_ax.set_xlabel(r'$\alpha$ [-]',fontsize=8)
        elif var == 'hfls':
            inset_ax.set_xlim(-3.5,3.5)
            if position == 7:
                pass
                # inset_ax.set_xlabel('LE [W m$^{-2}$]',fontsize=8)
        elif var == 'cLand':
            inset_ax.set_xlim(-0.75,1.15)
            if position == 8:
                pass
                # inset_ax.set_xlabel('C [gC m$^{-2}$]',fontsize=8)            

    ### No decimal for ticklabel 0
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
    
    extend='both'
    
    ### Plot two colorsbars                    
    if position == 0:
        cax = plt.axes([0.03, 0.67, 0.26, 0.025])           
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label='$\Delta$ Albedo [%]',fontsize=10,labelpad=15)
        cbar.ax.xaxis.set_label_position('top')

        ticks_above = {-25,-15,-5,5,15,25}
        ticks_below = {-30,-20,-10,-1,1,10,20,30}
        
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
        cax = plt.axes([0.5, 0.67, 0.32, 0.025])

        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )   
        
        ### Reduce fontsize
        cbar.set_label(label='$\Delta$ Albedo [%]',fontsize=10,labelpad=15)
        cbar.ax.tick_params(axis='x', labelsize=8)
        cbar.ax.xaxis.set_label_position('top')
        
        ticks_above = {-12.5,-7.5,-2.5,2.5,7.5,12.5}
        ticks_below = {-15,-10,-5,-1,1,5,10,15}

        for label in cbar.ax.get_xticklabels():
            # label_value = float(label.get_text())
            label_value = float(label.get_text().replace('−', '-'))  # Replace Unicode minus
            if label_value in ticks_above:
                label.set_va('top') 
                label.set_position((label.get_position()[0], 2.2)) # Move label above
            elif label_value in ticks_below:
                label.set_va('bottom')
                label.set_position((label.get_position()[0], -0.5))
    
    elif position == 3:
        cax = plt.axes([0.03, 0.36, 0.26, 0.025])
        label = 'Latent heat flux [W m$^{-2}$]'                
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label='$\Delta$ '+label,fontsize=10,labelpad=15)
        cbar.ax.xaxis.set_label_position('top')

        ticks_above = {-12.5,-7.5,-2.5,2.5,7.5,12.5}
        ticks_below = {-15,-10,-5,-0.5,0.5,5,10,15}
        
        for label in cbar.ax.get_xticklabels():
            # label_value = float(label.get_text())
            label_value = float(label.get_text().replace('−', '-'))  # Replace Unicode minus
            if label_value in ticks_above:
                label.set_va('top') 
                label.set_position((label.get_position()[0], 2.2)) # Move label above
            elif label_value in ticks_below:
                label.set_va('bottom')
                label.set_position((label.get_position()[0], -0.5))

    elif position == 5:
        ### left, bottom, width, height
        cax = plt.axes([0.5, 0.36, 0.32, 0.025])
        label = 'Latent heat flux [W m$^{-2}$]'
                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )   
        
        ### Reduce fontsize
        cbar.set_label(label='$\Delta$ '+label,fontsize=10,labelpad=15)
        cbar.ax.tick_params(axis='x', labelsize=8)
        cbar.ax.xaxis.set_label_position('top')
        
        ticks_above = {-6.25,-3.75,-1.25,1.25,3.75,6.25}
        ticks_below = {-7.5,-5,-2.5,-0.5,0.5,2.5,5,7.5}

        for label in cbar.ax.get_xticklabels():
            # label_value = float(label.get_text())
            label_value = float(label.get_text().replace('−', '-'))  # Replace Unicode minus
            if label_value in ticks_above:
                label.set_va('top') 
                label.set_position((label.get_position()[0], 2.2)) # Move label above
            elif label_value in ticks_below:
                label.set_va('bottom')
                label.set_position((label.get_position()[0], -0.5))
    
                
    elif position == 6:
        cax = plt.axes([0.03, 0.045, 0.26, 0.025])
        label = 'Carbon [gC m$^{-2}$]'             
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label='$\Delta$ '+label,fontsize=10,labelpad=15)
        cbar.ax.xaxis.set_label_position('top')

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


    elif position == 8:
        ### left, bottom, width, height
        cax = plt.axes([0.5, 0.045, 0.32, 0.025])
        label = 'Carbon [gC m$^{-2}$]'     
                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )   
        
        ### Reduce fontsize
        cbar.set_label(label='$\Delta$ '+label,fontsize=10,labelpad=15)
        cbar.ax.tick_params(axis='x', labelsize=8)
        cbar.ax.xaxis.set_label_position('top')
        
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
    
    ### Align y-axis with latitudes of map
    axs[position].set_extent([-180, 180, -60, 90], ccrs.PlateCarree())
    inset_ax.set_ylim(axs[position].get_ylim())

def call_plot(paper_part):             
    ### Decide which carbon pools to plot
    driver_1, driver_2, driver_3 = 'albedo','hfls','cLand'

    if paper_part == 'main':
        exp_ref_row_1, exp_pert_row_1 = 'a7cy', 'a7cy'
        exp_ref_row_2, exp_pert_row_2 = 'a7cy', 'a7en'
        exp_ref_row_3, exp_pert_row_3 = 'a7en', 'a7eo'

    elif paper_part == 'supplement':
        exp_ref_row_1, exp_pert_row_1 = 'a7en', 'a7en'
        exp_ref_row_2, exp_pert_row_2 = 'a7eo', 'a7eo'
        exp_ref_row_3, exp_pert_row_3 = 'a7eo', 'a7cy'    
        
    ## Plot drivers
    make_map(driver_1,first_year,last_year,exp_ref_row_1,exp_pert_row_1,0)
    make_map(driver_2,first_year,last_year,exp_ref_row_1,exp_pert_row_1,3)
    make_map(driver_3,first_year,last_year,exp_ref_row_1,exp_pert_row_1,6)
    make_map(driver_1,first_year,last_year,exp_ref_row_2,exp_pert_row_2,1)
    make_map(driver_2,first_year,last_year,exp_ref_row_2,exp_pert_row_2,4)
    make_map(driver_3,first_year,last_year,exp_ref_row_2,exp_pert_row_2,7)
    make_map(driver_1,first_year,last_year,exp_ref_row_3,exp_pert_row_3,2)
    make_map(driver_2,first_year,last_year,exp_ref_row_3,exp_pert_row_3,5)
    make_map(driver_3,first_year,last_year,exp_ref_row_3,exp_pert_row_3,8)

    if paper_part == 'main':
        ## Plot subplot titles
        axs[0].set_title('Reference (2071-2100 vs. 1981-2010)\nAlbedo',
                         fontsize=10,linespacing=1.75)
        axs[3].set_title('Latent heat flux',fontsize=10)
        axs[6].set_title('Carbon stored in land',fontsize=10)
        
        axs[1].set_title('Moderate CDR Ambition vs. Reference (2071-2100)\nAlbedo',
                         fontsize=10,linespacing=1.75)
        axs[4].set_title('Latent heat flux',fontsize=10)
        axs[7].set_title('Carbon stored in land',fontsize=10)

        axs[2].set_title('High vs. Moderate CDR Ambition (2071-2100)\nAlbedo',
                         fontsize=10,linespacing=1.75)                        
        axs[5].set_title('Latent heat flux',fontsize=10)
        axs[8].set_title('Carbon stored in land',fontsize=10)
        
    elif paper_part == 'supplement':
        axs[0].set_title('Moderate CDR Ambition (2071-2100 vs. 1981-2010)\nAlbedo',
                         fontsize=10,linespacing=1.75)
        axs[3].set_title('Latent heat flux',fontsize=10)
        axs[6].set_title('Carbon stored in land',fontsize=10)
        
        axs[1].set_title('High CDR Ambition (2071-2100 vs. 1981-2010)\nAlbedo',
                         fontsize=10,linespacing=1.75)
        axs[4].set_title('Latent heat flux',fontsize=10)
        axs[7].set_title('Carbon stored in land',fontsize=10)
        
        axs[2].set_title('High CDR Ambition vs. Reference (2071-2100)\nAlbedo',
                        fontsize=10,linespacing=1.75)
        axs[5].set_title('Latent heat flux',fontsize=10)
        axs[8].set_title('Carbon stored in land',fontsize=10)

    positions=[0,1,2,
               3,4,5,
               6,7,8]
    title_index=['a)','b)','c)',
                 'd)','e)','f)',
                 'g)','h)','i)']

    ### Loop through plot command, and adjust subplot titles
    for p, ti in zip(positions, title_index):
        axs[p].set_title(ti, loc='left',fontsize=9)

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
    plt.savefig('figures/maps_driver_'+first_year+'-'+last_year+'_'+paper_part+'_new_order_final.png',dpi=400)
    # plt.savefig('figures/maps_driver_'+first_year+'-'+last_year+'_'+paper_part+'_sign_flip.png',dpi=400)

### Set up figure    
figsize=(14,9)
plot_params = {'top':0.97, 'left':0.03,
               'right':0.96, 'bottom':0.1,
               'wspace':0.2, 'hspace':0.25}
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize,
                        sharex=True, sharey=True)
axs=axs.flatten()

call_plot('main')
# call_plot('supplement')
