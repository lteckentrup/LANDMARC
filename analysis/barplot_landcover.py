import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from matplotlib.gridspec import GridSpec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, required=True)
args = parser.parse_args()

### Assign variables
region=args.region

### Define pathway
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

### Read in gridarea 
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc')
ds_gridarea['lat'],ds_gridarea['lon'] = ds_mask['lat'],ds_mask['lon']

### Calculate area weighted average globally/regionally
def get_weighted_average(var,exp,biof,region):
    global pathwayIN
    global ds_gridarea

    if var == 'cropland':
        ### Read in variable
        if biof in ('biof','FFM'):
            ds_raw = xr.open_dataset(pathwayIN+exp+'_r1i1p1f1/crop_frac_'+exp+'-nn_years_'+biof+'.nc')
            da_raw = sum(ds_raw[var] for var in ds_raw.data_vars)                 
        else:
            da_raw = xr.open_dataset(pathwayIN+exp+'_r1i1p1f1/lu_frac_'+
                                     exp+'_years.nc')[var]
    elif var == 'area_share_afforestation':
        ### Read in variable
        da_raw = xr.open_dataset(pathwayIN+exp+'_r1i1p1f1/area_afforestation_'+
                                 exp+'_2015-2100.nc')[var]         
    else:
        ### Read in variable
        da_raw = xr.open_dataset(pathwayIN+exp+'_r1i1p1f1/lu_frac_'+
                                 exp+'_years.nc')[var]

    ### Invert latitude and align DataSets
    da = da_raw.reindex(lat=list(reversed(da_raw['lat']))).sel(time=slice('2070','2100')).mean(dim='time')
    da['lat'],da['lon'] = ds_mask['lat'],ds_mask['lon']
    
    ### Get land area
    da_landarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf']))
    da_land = da.where(~np.isnan(ds_mask['sftlf']))
    
    if region == 'Global':
        landsum = da_landarea.sum(dim=['lat','lon'])
        da_region = da_land
    elif region == 'Boreal':
        ### Get land area
        da_landarea = da_landarea.sel(lat=slice(50,90))
        landsum = da_landarea.sum(dim=['lat','lon'])
        da_region = da_land.sel(lat=slice(50,90))
    elif region == 'Tropical':
        ### Get land area
        da_landarea = da_landarea.sel(lat=slice(-30,30))
        landsum = da_landarea.sum(dim=['lat','lon'])
        da_region = da_land.sel(lat=slice(-30,30))       
    elif region == 'Temperate':
        ### Get land area
        da_landarea = da_landarea
        da_landarea = da_landarea.where(((da_landarea.lat >= -50) & (da_landarea.lat <= -30)) |
                                        ((da_landarea.lat >= 30) & (da_landarea.lat <= 50)),
                                        drop=True)
                        
        landsum = da_landarea.sum(dim=['lat','lon'])      
        da_region = da_land.where(((da_land.lat >= -50) & (da_land.lat <= -30)) |
                                  ((da_land.lat >= 30) & (da_land.lat <= 50)),
                                  drop=True)
                        
    ### Get weighted average
    da_region_weights = da_region * da_landarea
    da_region_weighted_mean = da_region_weights.sum(dim=['lat','lon'])/landsum

    return(da_region_weighted_mean.item()*100)

def get_data(exp,region):
    ### All experiments have natural, cropland and pasture
    sizes = [get_weighted_average('natural',exp,'',region),
             get_weighted_average('pasture',exp,'',region),
             get_weighted_average('cropland',exp,'',region)
             ]
    
    ### Other is the residual
    other = 100 - np.array(sizes).sum()
    
    ### Add residual
    sizes.append(other)
    sizes.append(get_weighted_average('cropland',exp,'biof',region))
    
    ### a7en and a7eo also includes A/F
    if exp in ('a7en','a7eo'):
        sizes.append(get_weighted_average('area_share_afforestation',exp,'',region))
                
    return(sizes)

def plot_barplot(axis,region):
    ### Set up DataFrame
    df = pd.DataFrame()
    df['Label'] = ['Natural','Pasture', 'Cropland', 'Other']
    a7cy = get_data('a7cy',region)
    a7en = get_data('a7en',region)
    a7eo = get_data('a7eo',region)
    
    ### Reference at top of horizontal barplot, High CDR Ambition at bottom
    df['High'] = a7eo[:4]
    df['Moderate'] = a7en[:4]
    df['Reference'] = a7cy[:4]
    
    ### Plot horizonal barplot with 4 main landcover / land-use types
    df.set_index('Label',inplace=True)
    colors = ['#44AA99', '#999933', '#DDCC77','#f1f1f1']
    axis = df.T.plot.barh(stacked=True,color=colors,ax=axis)
    
    ### Set up hatching
    ### AF
    hatch_AF_a7eo = a7eo[5]
    hatch_AF_a7en = a7en[5]

    ### High CDR Ambition
    axis.barh(
        y=0, 
        width=hatch_AF_a7eo,
        left=0,
        height=0.5,
        facecolor='none',
        edgecolor='k',
        hatch='////'
    )
    
    ### Moderate CDR Ambition
    axis.barh(
        y=1,
        width=hatch_AF_a7en,
        left=0,
        height=0.5,
        facecolor='none',
        edgecolor='k',
        hatch='////'
    )

    ### Biofuel
    hatch_biof_a7eo = a7eo[4]
    hatch_biof_a7en = a7en[4]
    hatch_biof_a7cy = a7cy[4]
    
    ### High CDR Ambition   
    axis.barh(
        y=0,
        width=hatch_biof_a7eo,
        left=df['High'].iloc[0]+df['High'].iloc[1],
        height=0.5,
        facecolor='none',
        edgecolor='k',
        hatch='\\\\\\\\'
    )
    
    ### Moderate CDR Ambition
    axis.barh(
        y=1,
        width=hatch_biof_a7en,
        left=df['Moderate'].iloc[0]+df['Moderate'].iloc[1],
        height=0.5,
        facecolor='none',
        edgecolor='k',
        hatch='\\\\\\\\'
    )
    
    ### Reference
    axis.barh(
        y=2,
        width=hatch_biof_a7cy,
        left=df['Reference'].iloc[0]+df['Reference'].iloc[1],
        height=0.5,
        facecolor='none',
        edgecolor='k',
        hatch='\\\\\\\\'
    )
    
    ### Hatching args
    for p in axis.patches:
        if p.get_hatch():
            p.set_linewidth(0)

    ### Remove spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    ### Set tick-labelsizes
    axis.tick_params(axis='both', labelsize=14)
    
    ### Set xlabel
    axis.set_xlabel('Landcover [%]',fontsize=15)
    
    ### Plot legend
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),
                frameon=False, ncol=2, fontsize=14)

fig=plt.figure(figsize=(6,3))
ax1=fig.add_subplot(1,1,1)

plot_barplot(ax1,region)

plt.tight_layout()
plt.savefig('figures/landcover/barplot_landcover_'+region+'.png', dpi=400)
