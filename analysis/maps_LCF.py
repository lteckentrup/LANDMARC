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

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5

### Set pathway where input files are located
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

### Read in gridarea 
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc')

def read_in_file(var,biof,exp):
    global pathwayIN
    global ds_mask
        
    ### Read in all natural vegetation covers
    if var == 'cropland' and biof in ('biof', 'FFM'):
        fname = (pathwayIN+exp+'_r1i1p1f1/crop_frac_'+exp+'-nn_years_'+biof+'.nc')
        ds_raw = xr.open_dataset(fname)
        da = sum(ds_raw[var] for var in ds_raw.data_vars)
        ds = da.to_dataset(name='cropland')
    elif var == 'area_share_afforestation':
        if exp == 'a7cy':
            fname=(pathwayIN+'a7en_r1i1p1f1/area_afforestation_a7en_2015-2100.nc')
            ds = xr.open_dataset(pathwayIN+'a7en_r1i1p1f1/area_afforestation_a7en_2015-2100.nc') 
            ds[var] = ds[var] * 0
        else:            
            fname=(pathwayIN+exp+'_r1i1p1f1/area_afforestation_'+exp+'_2015-2100.nc')
            ds = xr.open_dataset(pathwayIN+exp+'_r1i1p1f1/area_afforestation_'+
                                 exp+'_2015-2100.nc')      
    else:
        ### Read in variable
        fname=(pathwayIN+exp+'_r1i1p1f1/lu_frac_'+exp+'_years.nc')
        ds = xr.open_dataset(fname)
    
    ### Get percentage values
    ds[var] = ds[var] * 100
    
    ### Invert latitudes
    ds = ds.reindex(lat=list(reversed(ds['lat'])))
        
    ### Align latitudes and longitudes
    ds['lat'], ds['lon'] = ds_mask['lat'], ds_mask['lon']
        
    ### Mask ocean points
    da = ds[var].where(ds_mask['sftlf'] != 0, np.nan)
    return(da)

### Calculate area weighted average globally/regionally
def get_area_weighted_average(var,exp,biof,first_year,last_year):
    global pathwayIN
    global ds_gridarea

    ### Get data
    da = read_in_file(var,biof,exp).sel(time=slice(first_year,last_year)).mean(dim='time')
    
    ### Align latitudes and longitudes
    ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']      
 
    ### Get land area
    da_landarea = ds_gridarea['cell_area'].where(ds_mask['sftlf'] != 0, np.nan)
    landsum = da_landarea.sum(dim=['lat','lon'])  
    
    ### Get weighted average
    da_var_weights = da * da_landarea
    da_var_fldmean = da_var_weights.sum(dim=['lat','lon'])/landsum
    
    return(da_var_fldmean.item())

def get_data_map(var,biof,exp_ref,exp_pert,first_year,last_year):
    ### Get data: Reference (either historical or a reference scenario)
    if exp_ref == exp_pert:
        da_ref = read_in_file(var,biof,exp_ref).sel(time=slice('2015','2015')).mean(dim='time')
    else:
        da_ref = read_in_file(var,biof,exp_ref).sel(time=slice(first_year,last_year)).mean(dim='time')
    
    ### Get data: Experiment to test
    da_exp = read_in_file(var,biof,exp_pert).sel(time=slice(first_year,last_year)).mean(dim='time')
    
    ### Mask where both experiments are 0
    da_noC = (da_exp == 0) & (da_ref == 0)

    ### Set points to np.nan where the mask is True
    da_exp_noC = da_exp.where(~da_noC, np.nan)
    da_ref_noC = da_ref.where(~da_noC, np.nan)

    ### Calculate difference
    da_diff = da_exp_noC - da_ref_noC
    
    return(da_diff)

### Set up plot for maps
def make_map(var,biof,exp_ref,exp_pert,first_year,last_year,position):
    global ds_mask
    
    ### Get data
    da = get_data_map(var,biof,exp_ref,exp_pert,first_year,last_year)
    
    ### Get coordinates
    lat, lon = da.lat, da.lon
            
    ### Set levels for colorbar
    levels = [-50,-20,-10,-5,-2,-1,1,2,5,10,20,50]
    cmap='BrBG'

    ### Set colorbar: assign new colors to arrow tips so add two extra colors
    pal = sns.color_palette(cmap, len(levels)+2)
    cols = pal.as_hex()
    cols[int(len(levels)/2)] = '#e4e4e4'
    cmap = ListedColormap(cols[1:-1])
    cmap.set_under(cols[0])
    cmap.set_over(cols[-1])
    extend = 'both'
    
    ### Add cyclic point
    da, lon = add_cyclic_point(da, coord=lon)

    ### Boundary norm: set ncolors to ACTUAL number of levels (excl. first and last cmap val)
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=len(cols[1:-1]))            

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap, 
                                 norm=norm,zorder=1,
                                 transform=ccrs.PlateCarree())   

    ### Hatching for biochar and BECCS in biofuel panels
    if var == 'cropland' and biof in ('biof', 'FFM'):
        da_BECCS = xr.open_dataset(pathwayIN+'/'+exp_pert+
                                   '_r1i1p1f1/landcover/beccs_dom_frac_'+
                                   exp_pert+'.nc')['total'][0]
        da_BECCS['lat'], da_BECCS['lon'] = ds_mask['lat'], ds_mask['lon']
        
        ### Mask out not significant changes
        da_BECCS = da_BECCS.where(da_BECCS != 0, np.nan)
        
        ### Hatching in magenta where cLand was negative in Reference but is positive in High CDR
        cf = axs[position].contourf(
            da_BECCS.lon, da_BECCS.lat,
            da_BECCS,
            levels=[0.5, 1.5],
            hatches=['//////'],
            colors='none',
            transform=ccrs.PlateCarree(),
            zorder=2
        )
    
    ### Include coast lines                      
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
        
    ## Plot two colorsbars    
    if position == 10:
        cax = plt.axes([0.26, 0.05, 0.5, 0.025])
        label = 'Landcover [%]'                
        cbar = fig.colorbar(p, 
                    cax=cax, 
                    ticks=levels, 
                    orientation='horizontal',
                    extend=extend, 
                    )
        cbar.set_label(label='$\Delta$ '+label,fontsize=10)
                   
    ### Reduce fontsize of coordinate labels
    axs[position].tick_params(axis='both', labelsize=8)     
    if exp_ref == exp_pert:
        height_perc = '22%'
    elif exp_ref == 'a7cy' and exp_ref != exp_pert:
        height_perc = '22%'
    else:
        height_perc = '15%'
          
    inset_ax = inset_axes(axs[position],
                          width="18%",
                          height=height_perc,
                          loc='lower left',
                          bbox_to_anchor=(0.06, 0.11, 1, 1),
                          bbox_transform=axs[position].transAxes)
    
    ### Get historical and end of century averages
    ### AF doesn't have values for Reference
    ### Reference
    if var == 'area_share_afforestation':
        a7cy_hist = 0
        a7cy = 0
    else:
        a7cy_hist = get_area_weighted_average(var,'a7cy',biof,'2015','2015')
        a7cy = get_area_weighted_average(var,'a7cy',biof,first_year,last_year)
    
    ### Moderate CDR Ambition
    a7en_hist = get_area_weighted_average(var,'a7en',biof,'2015','2015')
    a7en = get_area_weighted_average(var,'a7en',biof,first_year,last_year)
    
    ### High CDR Ambition
    a7eo_hist = get_area_weighted_average(var,'a7eo',biof,'2015','2015')
    a7eo = get_area_weighted_average(var,'a7eo',biof,first_year,last_year)
    
    ### Set up barplots
    if exp_ref == exp_pert:
        categories = ['High', 'Mod.','Ref.']
        colors = ['#00678a','#e6a176','#3e3e3e']
        values = [a7eo-a7eo_hist,
                  a7en-a7en_hist,
                  a7cy-a7cy_hist]
    elif exp_ref == 'a7cy':
        categories = ['High-\nRef.', 'Mod.-\nRef.']
        colors = ['#00678a','#e6a176']
        values = [a7eo-a7cy,
                  a7en-a7cy]
    else:
        categories = ['High-\nMod.']
        colors = ['#c0affb']
        values = [a7eo-a7en]

    ### Plot horizontal barplot
    inset_ax.barh(categories, values, color=colors)

    ### Inset axes
    inset_ax.set_xlabel('$\Delta$ Landcover [%]', fontsize=5, labelpad=-0.25)
    inset_ax.tick_params(axis='both', which='major', labelsize=5)
    
    ### Remove spines
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['top'].set_visible(False)
    
    ### Set limits for axes
    if exp_ref == exp_pert:
        inset_ax.set_xlim(-9,9)
        inset_ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    elif exp_ref == 'a7cy':
        inset_ax.set_xlim(-6,9.5)
        inset_ax.set_ylim(-0.75,2)
        inset_ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    else:
        inset_ax.set_xlim(-1.5,2)
        inset_ax.set_ylim(-0.8,1)
        inset_ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
        
### Set up figure
figsize=(15,12)
plot_params = {'top':0.95, 'left':0.04,
               'right':0.975, 'bottom':0.1,
               'wspace':0.08, 'hspace':0.2} 
    
fig, axs = plt.subplots(nrows=5,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize,
                        sharex=True, sharey=True)

axs=axs.flatten()

### Set up multipanel plot
def LCF_panel_plot(paper_part):
    global fig
    global axs
    global plot_params
    
    ### Set first and last year of future projection
    first_year='2070'
    last_year='2100'
    
    ### Set up short names for variables
    var_short_names = ['natural','pasture','cropland','cropland','area_share_afforestation']
    
    ### Define type of management
    management = ['','','','biof','']
    
    ### Set up plot locations
    pos_col1=[0,3,6,9,12]
    pos_col2=[1,4,7,10,13]
    pos_col3=[2,5,8,11,14]
    
    ### Number panels
    title_col1=['a)','d)','g)','j)','m)']
    title_col2=['b)','e)','h)','k)','n)']
    title_col3=['c)','f)','i)','l)','o)']
    
    ### Main or supplmentary figure?
    if paper_part == 'main':
        var_titles_col1 = ['Reference (2071-2100 vs. 2015)\nNatural vegetation',
                           'Pasture','Cropland','Biofuel','Afforestation/ Reforestation']
        var_titles_col2 = ['Moderate CDR Ambition vs. Reference (2071-2100)\nNatural vegetation',
                           'Pasture','Cropland','Biofuel','Afforestation/ Reforestation']
        var_titles_col3 = ['High vs. Moderate CDR Ambition (2071-2100)\nNatural vegetation',
                           'Pasture','Cropland','Biofuel','Afforestation/ Reforestation']
        ref_col1, exp_col1 = 'a7cy', 'a7cy'
        ref_col2, exp_col2 = 'a7cy', 'a7eo'
        ref_col3, exp_col3 = 'a7en', 'a7eo'
        
    else:
        var_titles_col1 = ['Moderate CDR Ambition (2071-2100 vs. 2015)\nNatural vegetation',
                           'Pasture','Cropland','Biofuel','Afforestation/ Reforestation']
        var_titles_col2 = ['High CDR Ambition (2071-2100 vs. 2015)\nNatural vegetation',
                           'Pasture','Cropland','Biofuel','Afforestation/ Reforestation']
        var_titles_col3 = ['High CDR Ambition vs. Reference (2071-2100)\nNatural vegetation',
                           'Pasture','Cropland','Biofuel','Afforestation/ Reforestation']
        ref_col1, exp_col1 = 'a7en', 'a7en'
        ref_col2, exp_col2 = 'a7eo', 'a7eo'
        ref_col3, exp_col3 = 'a7eo', 'a7cy'
                    
    ### Loop through plot command, and adjust subplot titles for each column
    for vars, man, t_c1, vt_c1, pos_c1 in zip(var_short_names, management, 
                                           title_col1,var_titles_col1, pos_col1):
        make_map(vars,man,ref_col1,exp_col1,first_year,last_year,pos_c1)
        axs[pos_c1].set_title(t_c1, loc='left')
        axs[pos_c1].set_title(vt_c1, linespacing=1.75)
        axs[pos_c1].yaxis.set_visible(True)
        
    for vars, man, t_c2, vt_c2, pos_c2 in zip(var_short_names, management, 
                                              title_col2,var_titles_col2, pos_col2):
        make_map(vars,man,ref_col2,exp_col2,first_year,last_year,pos_c2)
        axs[pos_c2].set_title(t_c2, loc='left')
        axs[pos_c2].set_title(vt_c2, linespacing=1.75)
        axs[pos_c2].yaxis.set_visible(True)
        axs[pos_c2].tick_params(axis='y', labelleft=False)
        
    for vars, man, t_c3, vt_c3, pos_c3 in zip(var_short_names, management, title_col3,
                                              var_titles_col3, pos_col3):
        make_map(vars,man,ref_col3,exp_col3,first_year,last_year,pos_c3)
        axs[pos_c3].set_title(t_c3, loc='left')
        axs[pos_c3].set_title(vt_c3, linespacing=1.75)
        axs[pos_c3].yaxis.set_visible(True)
        axs[pos_c3].tick_params(axis='y', labelleft=False)
    
    ### Show ticklabels for x-axis for three bottom panels
    for p in (12,13,14):
        axs[p].xaxis.set_visible(True)
    
    ### Show ticks but not ticklables for remaining x-axes
    for p in (0,1,2,3,4,5,6,7,8,9,10,11):
        axs[p].xaxis.set_visible(True)
        axs[p].tick_params(axis='x', labelbottom=False)

    ### Show ticks but not ticklables for y-axis for all except left column panels  
    for p in (1,2,4,5,7,8,10,11,13,14):
        axs[p].yaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelleft=False)
    
    plt.subplots_adjust(**plot_params)
    plt.savefig('figures/landcover/maps_LCF_LMT_'+paper_part+'.png',dpi=400)

LCF_panel_plot('main')
# LCF_panel_plot('supplement')
