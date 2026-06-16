import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import cartopy
import cartopy.crs as ccrs
from  matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### Set pathway where input files are located
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

### Read in gridarea 
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc')

ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']
da_gridarea = ds_gridarea['cell_area'].where(ds_mask['sftlf'] != 0, np.nan)

### Set up plot for maps
def make_map(var,exp_ref,exp_pert,first_year,last_year,position):
    global ds_mask
    
    ### Get flip sign
    da_flip = xr.open_dataset(pathwayIN+
                              '/robustness_maps/'+first_year+'-'+last_year+
                              '/landcover/'+var+'/'+var+'_EC-Earth3-CC_ssp245_'+
                              exp_ref+'_'+exp_pert+'_gr_sign_flip.nc')['flip']
    da_flip['lat'], da_flip['lon'] = ds_mask['lat'], ds_mask['lon']

    lat, lon = da_flip['lat'], da_flip['lon']
    
    ### Mask out not significant changes
    da_flip_cyclic, lon = add_cyclic_point(da_flip, coord=da_flip.lon)

    ### Set colorbar: assign new colors to arrow tips so add two extra colors
    levels = [0,1,2]
    cols = ['#912a23', '#118e82']

    ### cmap based on actual number of levels
    cmap = ListedColormap(cols)
    
    ### Set bounds
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=cmap.N)
    
    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da_flip_cyclic, cmap=cmap, norm=norm,zorder=1)
    
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
    
    ### Get percentage of landarea with sign. positive and negative values
    masks = {
        'pos': (da_flip > 0),
        'neg': (da_flip < 0)
    }

    percs = {
        key: 100 * ((da_gridarea * mask).sum(dim=['lat', 'lon']) /
                     da_gridarea.sum(dim=['lat', 'lon']))
        for key, mask in masks.items()
    }

    ### Round results
    perc_sign = percs['pos'] + percs['neg']
    perc_sign_pos = 100*(percs['pos']/perc_sign)
    perc_sign_neg = 100*(percs['neg']/perc_sign)
    
    ### Print results percentages in bottom left corner
    axs[position].text(
        0.02, 0.02,
        f'Total: {perc_sign.item():.2f}%\n'
        f'Pos: {perc_sign_pos.item():.2f}%\n'
        f'Neg: {perc_sign_neg.item():.2f}%',
        transform=axs[position].transAxes,
        fontsize=8,
        color='black',
        bbox=dict(facecolor='white',
                  alpha=0,
                  edgecolor='none')
        )   

### Set up figures    
figsize=(14,7)
plot_params = {'top':0.96, 'left':0.03,
               'right':0.96, 'bottom':0.05,
               'wspace':0.05, 'hspace':0.05}
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=figsize,
                        sharex=True, sharey=True)

axs=axs.flatten()

### Set up multi-panel plot
def LCF_panel_plot(paper_part):
    global fig
    global axs
    global plot_params
    
    ### Set first and last year of future projection
    first_year='2071'
    last_year='2100'
    
    ### Set up short names for variables
    var_short_names = ['natural','pasture','cropland','cropland','area_share_afforestation']
    
    ### Set up plot locations
    pos_MOD_REF=[0,3,6]
    pos_HIGH_REF=[1,4,7]
    pos_HIGH_MOD=[2,5,8]
    
    ### Number panels
    title_MOD_REF=['a)','d)','g)']
    title_HIGH_REF=['b)','e)','h)']
    title_HIGH_MOD=['c)','f)','i)']
    
    ### Set up plot titles 
    var_titles_REF = ['Moderate CDR Ambition vs. Reference (2071-2100)\nNatural vegetation',
                      'Pasture',
                      'Cropland']
    var_titles_HIGH_REF = ['High CDR Ambition vs. Reference (2071-2100)\nNatural vegetation',
                          'Pasture',
                          'Cropland']
    var_titles_HIGH_MOD = ['High vs. Moderate CDR Ambition (2071-2100)\nNatural vegetation',
                           'Pasture',
                           'Cropland']
                        
    ### First row: Natural vegetation
    for vars, t_R, vt_R, pos_R in zip(var_short_names,
                                           title_MOD_REF,var_titles_REF, pos_MOD_REF):
        make_map(vars,'a7cy','a7en',first_year,last_year,pos_R)
        axs[pos_R].set_title(t_R, fontsize=9, loc='left')
        axs[pos_R].set_title(vt_R, fontsize=9, linespacing=1.75)
        axs[pos_R].yaxis.set_visible(True)
        axs[pos_R].tick_params(axis='x', labelsize=8)
        axs[pos_R].tick_params(axis='y', labelsize=8)
    
    ### Second row: Pasture
    for vars, t_MR, vt_MR, pos_MR in zip(var_short_names,
                                              title_HIGH_REF,var_titles_HIGH_REF, pos_HIGH_REF):
        make_map(vars,'a7cy','a7eo',first_year,last_year,pos_MR)
        axs[pos_MR].set_title(t_MR, fontsize=9, loc='left')
        axs[pos_MR].set_title(vt_MR, fontsize=9, linespacing=1.75)
        axs[pos_MR].yaxis.set_visible(True)
        axs[pos_MR].tick_params(axis='y', labelleft=False)
        axs[pos_MR].tick_params(axis='x', labelsize=8)
        axs[pos_MR].tick_params(axis='y', labelsize=8)
                
    ### Third row: Cropland 
    for vars, t_HM, vt_HM, pos_HM in zip(var_short_names,
                                              title_HIGH_MOD,var_titles_HIGH_MOD, pos_HIGH_MOD):
        make_map(vars,'a7en','a7eo',first_year,last_year,pos_HM)
        axs[pos_HM].set_title(t_HM, fontsize=9, loc='left')
        axs[pos_HM].set_title(vt_HM, fontsize=9, linespacing=1.75)
        axs[pos_HM].yaxis.set_visible(True)
        axs[pos_HM].tick_params(axis='y', labelleft=False)
        axs[pos_HM].tick_params(axis='x', labelsize=8)
        axs[pos_HM].tick_params(axis='y', labelsize=8)
    
    ### Show bottom ticks and ticklabels bottom    
    for p in (6,7,8):
        axs[p].xaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelsize=8)
    
    ### Show bottom ticks but not labels (except bottom)
    for p in (0,1,2,3,4,5):
        axs[p].xaxis.set_visible(True)
        axs[p].tick_params(axis='x', labelsize=8, labelbottom=False)

    ### Show yticks but not labels (except left column)
    for p in (1,2,4,5,7,8):
        axs[p].yaxis.set_visible(True)
        axs[p].tick_params(axis='y', labelsize=8, labelleft=False)

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
        
    plt.subplots_adjust(**plot_params)
    plt.savefig('figures/landcover/maps_LCF_LMT_'+paper_part+'_sign_flip.png',dpi=400)


LCF_panel_plot('main')
# LCF_panel_plot('supplement')
