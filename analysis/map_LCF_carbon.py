import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy
import cartopy.crs as ccrs
from  matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter

### Set pathway where input files are located
pathwayIN=define pathway where data are stored

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

### Read in gridarea 
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc')
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

def get_data(var):
    global pathwayIN
    global ds_mask
    
    if var in ('natural','cropland','pasture'):
        pass
    else:
        ID = var_to_ID_dict[var]
    
    ### Read in all natural vegetation covers
    if var in ('natural','cropland','pasture'):
        fname=(pathwayIN+'a7cy_r1i1p1f1/lu_frac_a7cy_years.nc')
        ds = xr.open_dataset(fname).sel(time=slice('2015','2015'))
        ds[var] = ds[var] * 100
        ds = ds.reindex(lat=list(reversed(ds['lat'])))
        da = ds[var]
    else:
        fname=(pathwayIN+'/a766_r1i1p1f1/carbon/'+var+'/'+var+'_'+
               ID+'_EC-Earth3-CC_historical_r1i1p1f1_gr_1850-2014.nc')
        da = xr.open_dataset(fname)[var].sel(time=slice('1981','2010')).mean(dim='time')
    
    print(fname)    
    ### Set latitudes and longitudes (? precision in a6xx slightly off maybe?)
    da['lat'] = ds_mask['lat']
    da['lon'] = ds_mask['lon']
        
    ### Mask ocean points
    da_land = da.where(ds_mask['sftlf'] != 0, np.nan)    
    
    if var in ('natural','cropland','pasture'):
        da_land[0].to_dataset(name=var).to_netcdf(var+'.nc')
        return(da_land[0])
    else:
        da_land.to_dataset(name=var).to_netcdf(var+'.nc')
        return(da_land)
    
### Calculate area weighted average globally/regionally
def get_weighted_average(da,var):
    global pathwayIN
    global ds_gridarea
 
    ### Get land area
    da_landarea = ds_mask.sftlf.values * ds_gridarea['cell_area']
    
    ### Get weighted sum
    da_var_weights = da * da_landarea
    da_var_fldsum = da_var_weights.sum(dim=['lat','lon'])
    
    da_var_fldsum.to_dataset(name=var).to_netcdf(var+'_fldsum.nc')

    if var in ('natural','cropland','pasture'):
        return(da_var_fldsum.item()/1e+10)
    else:
        return(da_var_fldsum.item()/1e+12)

def get_zonal_sum(da,var):
    ### Get zonal sum - weighted by gridarea and then sum across longitudes
    da_weighted = da * da_gridarea
    da_zonal = da_weighted.sum(dim='lon')

    if var in ('natural','cropland','pasture'):
        return(da_zonal/1e10)
    else:
        return(da_zonal/1e+12)
            
### Set up plot for maps
def make_map(var,position):
    global ds_mask
    
    ### Get data
    da = get_data(var)
    da_zonal = get_zonal_sum(da,var)
    
    da_zonal.to_dataset(name=var).to_netcdf(var+'_zonsum.nc')
    
    ### Get coordinates  
    lat, lon = da.lat, da.lon

    ### Set up colormap
    cols = ['#e4e4e4', '#FDE725FF', '#BBDF27FF', '#7AD151FF',
            '#43BF71FF', '#22A884FF', '#21908CFF', '#2A788EFF',
            '#35608DFF', '#414487FF', '#482576FF', '#440154FF', 
            '#440154FF', '#440154FF']
                          
    ### Set levels for colorbar
    if var in ('natural','cropland','pasture'):
        levels = [0,1,2,3,4,5,10,20,30,40,50,100]
        extend = 'neither'
        cmap = ListedColormap(cols)
    else:
        if var in ('cLand','cSoil'):
            levels = [0,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]
        elif var == 'cVeg':
            levels = [0,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10]

        ### Set colorbar: assign new colors to arrow tips so add two extra colors
        cmap = ListedColormap(cols[:-1])
        cmap.set_over(cols[-1])
        extend = 'max'

    ### Add cyclic point
    da, lon = add_cyclic_point(da, coord=lon)

    ### Boundary norm: set ncolors to ACTUAL number of levels (excl. first and last cmap val)
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=len(cols[1:-1]))            

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap, 
                                 norm=norm,zorder=1,
                                 transform=ccrs.PlateCarree())   
    
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
    
    ### Plot zonal sum to the right
    ### Set up axis
    inset_zonal = inset_axes(axs[position],
                             width='10%',
                             height='100%',
                             loc='center right',
                             bbox_to_anchor=(0.15, 0, 1, 1),
                             bbox_transform=axs[position].transAxes)
        
    ### Plot zonal sum          
    inset_zonal.plot(da_zonal.values,
                     da_zonal.lat,color='#00678a')

    ### Remove yticks
    inset_zonal.set_yticklabels([])
    
    ### Set fontsize for x ticks
    inset_zonal.tick_params(axis='x', labelsize=9)
    
    ### Set xlabel
    if position in (0,2,4):
        inset_zonal.set_xlim(0,8500)
    else:
        inset_zonal.set_xlim(0,29)
        
    if position == 4:
        inset_zonal.set_xlabel('Area [Mha]',fontsize=9)
    elif position == 5:
        inset_zonal.set_xlabel('Carbon [PgC]',fontsize=9)
    else:
        inset_zonal.set_xlabel('')

    ### Drop spines
    inset_zonal.spines['right'].set_visible(False)
    inset_zonal.spines['top'].set_visible(False)
    
    ### Plot global values in bottom panels
    if position in (4,5):
        ### Reduce fontsize of coordinate labels
        height_perc = '22%'
        
        ### Set up axis
        inset_barplot = inset_axes(axs[position],
                                width='12%',
                                height=height_perc,
                                loc='lower left',
                                bbox_to_anchor=(0.115, 0.2, 1, 1),
                                bbox_transform=axs[position].transAxes)
        
        ### Landcover left
        if position == 4:
            ### Get three landcover types
            da_natural = get_data('natural')
            da_cropland = get_data('cropland')
            da_pasture = get_data('pasture')
            
            ### Get area-weighted sums
            natural = get_weighted_average(da_natural,'natural')
            cropland = get_weighted_average(da_cropland,'cropland')
            pasture = get_weighted_average(da_pasture,'pasture')
            
            ### Define labels, colors and value list
            categories = ['Natural', 'Pasture', 'Cropl.']
            value_list = [natural,cropland,pasture]
            colors = ['#44AA99', '#999933', '#DDCC77']
            
            inset_barplot.set_xlabel('Area [Mha]', fontsize=7)
            
        ### Carbon pools right
        elif position == 5:
             ### Get three carbon pools
            da_cLand = get_data('cLand')
            da_cSoil = get_data('cSoil')
            da_cVeg = get_data('cVeg')
            
            ### Get area-weighted sums
            cLand = get_weighted_average(da_cLand,'cLand')
            cSoil = get_weighted_average(da_cSoil,'cSoil')
            cVeg = get_weighted_average(da_cVeg,'cVeg')
            
            ### Define labels, colors and value list
            categories = ['cLand', 'cSoil', 'cVeg']
            value_list = [cLand,cSoil,cVeg]
            colors = ['#44AA99', '#999933', '#DDCC77']

            inset_barplot.set_xlabel('Carbon [PgC]', fontsize=7)
        
        ### Plot horizontal barplot
        inset_barplot.barh(categories[::-1], value_list[::-1], color=colors)
        
        ### Drop spines
        inset_barplot.spines['right'].set_visible(False)
        inset_barplot.spines['top'].set_visible(False)
        
        ### Set label fontsize
        inset_barplot.tick_params(axis='x', labelsize=7)
        inset_barplot.tick_params(axis='y', labelsize=7)

    ### Plot two colorsbars
    if position == 4:
        ### left, bottom, width, height
        cax = plt.axes([0.05, 0.08, 0.401, 0.025])
        label = 'Landcover [%]'
                        
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )   
        
        ### Reduce fontsize
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.set_label(label=label,fontsize=9)  
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    
    elif position == 5:
        cax = plt.axes([0.53, 0.08, 0.401, 0.025])
        label = 'Carbon stored in vegetation [kgC m$^{-2}$]'                
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )
        cbar.set_label(label=label,fontsize=9)
        cbar.ax.tick_params(axis='x', labelsize=8) 
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        
    elif position == 3:
        cax = plt.axes([0.53, 0.08, 0.401, 0.025])
        label = 'Carbon stored in land or soil [kgC m$^{-2}$]'                
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(axis='x', labelsize=8) 
        
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        
        cbar.set_label(label=label,fontsize=9)

### Set up figure
figsize=(9,7)
plot_params = {'top':0.95, 'left':0.05,
               'right':0.91, 'bottom':0.2,
               'wspace':0.2, 'hspace':0.25}
fig, axs = plt.subplots(nrows=3,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize)
axs=axs.flatten()

### Plot map panels
make_map('natural',0)
make_map('cropland',2)
make_map('pasture',4)

make_map('cLand',1)
make_map('cSoil',3)
make_map('cVeg',5)

### Set titles
positions=[0,1,
           2,3,
           4,5]
title_index=['a)','b)',
             'c)','d)',
             'e)','f)']
title_names=['Natural', 'Carbon stored in land',
             'Cropland', 'Carbon stored in soil',
             'Pasture', 'Carbon stored in vegetation']

### Loop through plot command, and adjust subplot titles
for p, ti, tn in zip(positions, title_index, title_names):
    axs[p].set_title(ti, loc='left',fontsize=9)
    axs[p].set_title(tn, fontsize=9)

### Adjust ticklabels on axes
for p in (0,2,4):
    ### Show ticklabels left
    axs[p].yaxis.set_visible(True)

for p in (4,5):
    ### Show ticklabels bottom
    axs[p].xaxis.set_visible(True)
    axs[p].tick_params(axis='x',labelsize=9)

for p in (1,3,5):
    axs[p].yaxis.set_visible(True)
    axs[p].tick_params(axis='y', labelleft=False,labelsize=9)

plt.subplots_adjust(**plot_params)
plt.savefig('figures/maps_LCF_carbon_historical.png',dpi=400)
