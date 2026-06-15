import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.mpl.geoaxes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)
parser.add_argument('--plot_type', type=str, required=True)
args = parser.parse_args()

### Assign variables
region=args.region
first_year=args.first_year
last_year=args.last_year
plot_type=args.plot_type

### Define pathway
pathwayIN='/gpfs/scratch/bsc32/bsc032352/LANDMARC/'

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

### Read in gridarea 
ds_gridarea = xr.open_dataset(pathwayIN+'aux/gridarea.nc')
ds_gridarea['lat'], ds_gridarea['lon'] = ds_mask['lat'], ds_mask['lon']

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
        da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon'] 
                
        ### Cumulative sum for fco2fossub
        if var == 'fco2fossub':
            da = da.cumsum(dim='time')

        da_final = da.sel(time=slice(first_year,last_year)).where(ds_mask['sftlf'] != 0, np.nan)   
        
        ### Return dataset
        return(da_final)

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
        da_final = da * 0
            
        ### Return dataset
        return(da_final.sel(time=slice(first_year,last_year)))

### Calculate area weighted average globally/regionally
def get_weighted_average(region,exp_id,realisation,var,first_year,last_year,plot_type):
    global pathwayIN
    global ds_gridarea
    global ds_mask

    ### Get data
    da = get_data(var,exp_id,realisation,first_year,last_year)
    
    ### Get area weighted sum
    ### Multply with gridarea
    da_var_gridarea = da * ds_gridarea['cell_area']
    da_var_gridarea_mask = da_var_gridarea.where(~np.isnan(ds_mask['sftlf']))

    if plot_type == 'average':
        da_gridarea = ds_gridarea['cell_area'].where(ds_mask['sftlf'] == 1, np.nan)
        landsum = da_gridarea.sum(dim=['lat','lon']) 
        da_weighted = (da_var_gridarea_mask.sum(dim=['lat', 'lon'])/landsum) * 1000
    else:
        if region == 'Global':
            da_weighted = da_var_gridarea_mask.sum(dim=['lat', 'lon'])/1e+12
        elif region == 'Tropical':
            da_weighted = da_var_gridarea_mask.sel(lat=slice(-30,30)).sum(dim=['lat', 'lon'])/1e+12
        elif region == 'Boreal':
            da_weighted = da_var_gridarea_mask.sel(lat=slice(50,90)).sum(dim=['lat', 'lon'])/1e+12
        elif region == 'Temperate':
            da_var = da_var_gridarea_mask.where(((da_var_gridarea_mask.lat >= -50) & (da_var_gridarea_mask.lat <= -30)) |
                                                ((da_var_gridarea_mask.lat >= 30) & (da_var_gridarea_mask.lat <= 50)),
                                                drop=True)
            da_weighted = da_var.sum(dim=['lat', 'lon'])/1e+12

    ### Create dataframe
    df = pd.DataFrame() 
    df['time'] = da.time.dt.year         
    
    ### Cumulative sum for nbp, fLuc, fco2fossub and foc2antt      
    df[var] = da_weighted.values
    
    ### Set time as index
    return(df.set_index('time'))

def make_dataframe(region,exp_id,realisation,first_year,last_year,plot_type):
    ### Create dataframe
    df = pd.concat([get_weighted_average(region,exp_id,realisation,'cVeg',first_year,last_year,plot_type),
                    get_weighted_average(region,exp_id,realisation,'cLitter',first_year,last_year,plot_type),
                    get_weighted_average(region,exp_id,realisation,'cSoil',first_year,last_year,plot_type),
                    get_weighted_average(region,exp_id,realisation,'cBECCS',first_year,last_year,plot_type),
                    get_weighted_average(region,exp_id,realisation,'cBiochar',first_year,last_year,plot_type),
                    get_weighted_average(region,exp_id,realisation,'fco2fossub',first_year,last_year,plot_type)],
                   axis=1)

    return(df)

def calc_diff(region,exp_ref,exp_pert,realisation,first_year,last_year,plot_type):
    ### Get reference: either historical or reference/ LMT end of century
    if exp_ref in ('a3bh','a3nm','a3o0','a766'):
        first_year_ref, last_year_ref = '1981','2010'
    else:
        first_year_ref, last_year_ref = first_year, last_year
    
    df_ref = make_dataframe(region,exp_ref,realisation,first_year_ref,last_year_ref,plot_type)
    
    ### Get dataframe for LMT experiment
    df_exp = make_dataframe(region,exp_pert,realisation,first_year,last_year,plot_type)
    
    if exp_ref in ('a3bh','a3nm','a3o0','a766'):
        df_diff = df_exp - df_ref.mean()
    else:
        df_diff = df_exp - df_ref
        
    return(df_diff)
       
def get_ens_dataframe(region,exp_ref,exp_pert,first_year,last_year,plot_type):
    ### Prepare dataframe for plot    
    df_merge = pd.concat([calc_diff(region,exp_ref,exp_pert,'r1i1p1f1',first_year,last_year,plot_type),
                          calc_diff(region,exp_ref,exp_pert,'r2i1p1f1',first_year,last_year,plot_type),
                          calc_diff(region,exp_ref,exp_pert,'r3i1p1f1',first_year,last_year,plot_type)])

    return(df_merge)

def make_barplot(region,legend_bool,first_year,last_year,plot_type):
    ### Get dataframes    
    # df_a7cy = get_ens_dataframe(region,'a766','a7cy',first_year,last_year,plot_type)
    # df_a7en = get_ens_dataframe(region,'a7cy','a7en',first_year,last_year,plot_type)
    # df_a7eo = get_ens_dataframe(region,'a7en','a7eo',first_year,last_year,plot_type)
    df_a7cy = get_ens_dataframe(region,'a766','a7cy',first_year,last_year,plot_type)
    df_a7en = get_ens_dataframe(region,'a766','a7en',first_year,last_year,plot_type)
    df_a7eo = get_ens_dataframe(region,'a766','a7eo',first_year,last_year,plot_type)

    ### Include identifier
    df_a7cy['Experiment'] = 'a7cy'
    df_a7en['Experiment'] = 'a7en'
    df_a7eo['Experiment'] = 'a7eo'
    
    ### Merge and melt for grouped boxplots
    df_merge = pd.concat([df_a7cy,df_a7en,df_a7eo])
    df_melted = df_merge.melt(id_vars=['Experiment'], 
                              value_vars=['cVeg','cLitter','cSoil',
                                          'cBECCS', 'cBiochar', 'fco2fossub'],
                              var_name='Variable', value_name='Value')
    
    ### Set up figure
    fig=plt.figure(figsize=(5.5,5))
    axis=fig.add_subplot(1,1,1)
    
    ### Remove white space
    plt.subplots_adjust(top=0.94, bottom=0.1,
                        right=0.99, left=0.13)

    ### Plot boxplots
    axis = sns.boxplot(x='Variable', y='Value',
                       hue='Experiment', data=df_melted,
                       palette=['tab:grey','#e6a176','#00678a'],
                       showfliers=False,ax=axis)
                  
    ### Remove spines                     
    sns.despine(ax=axis)

    ### Set xticks on the bottom axis
    axis.set_xticks([0, 1, 2, 3, 4, 5])
    axis.set_xticklabels(['$\mathit{cVeg}$',
                          '$\mathit{cLitter}$',
                          '$\mathit{cSoil}$',
                          '$\mathit{cBECCS}$',
                          '$\mathit{cBiochar}$',
                          r'$\mathit{Fossil\ fuel}$' + '\n' + r'$\mathit{substitution}$'
                          ])
    axis.set_xlabel('')
    
    ### Custom label
    if plot_type == 'average':
        label = '$\Delta$ Carbon Pool [gC m$^{-2}$]'
    else:
        label = '$\Delta$ Carbon Pool [PgC]'
    
    if region == 'Temperate':
        labelpad=17
    
    if region == 'Temperate': 
        axis.set_ylabel(label,fontsize=10,labelpad=labelpad)
    else:
        axis.set_ylabel('')
    
    if legend_bool:
        ### Remove frame and title
        axis.legend(frameon=False,loc='center right')
        axis.legend_.set_title(None)
    else:
        axis.legend_.remove()
    
    axis.axhline(0,color='tab:grey',lw=0.5)
    
    if region == 'Global':
        axis.set_title('a)', loc='left')
        axis.set_title('Global') 
    elif region == 'Temperate':
        axis.set_title('c)', loc='left')
        axis.set_title('Temperate')
    elif region == 'Tropical':
        axis.set_title('d)', loc='left')
        axis.set_title('Tropical')   
                       
    plt.savefig('figures/carbon/'+region+'_'+first_year+'-'+last_year+'_'+plot_type+'.png',dpi=400)
    
def make_barplot_break(region,legend_bool,first_year,last_year,plot_type):
    ### Get dataframes    
    # df_a7cy = get_ens_dataframe(region,'a766','a7cy',first_year,last_year,plot_type)
    # df_a7en = get_ens_dataframe(region,'a7cy','a7en',first_year,last_year,plot_type)
    # df_a7eo = get_ens_dataframe(region,'a7en','a7eo',first_year,last_year,plot_type)
    df_a7cy = get_ens_dataframe(region,'a766','a7cy',first_year,last_year,plot_type)
    df_a7en = get_ens_dataframe(region,'a766','a7en',first_year,last_year,plot_type)
    df_a7eo = get_ens_dataframe(region,'a766','a7eo',first_year,last_year,plot_type)

    ### Include identifier
    df_a7cy['Experiment'] = 'a7cy'
    df_a7en['Experiment'] = 'a7en'
    df_a7eo['Experiment'] = 'a7eo'
    
    ### Merge and melt for grouped boxplots
    df_merge = pd.concat([df_a7cy,df_a7en,df_a7eo])
    df_melted = df_merge.melt(id_vars=['Experiment'], 
                              value_vars=['cVeg','cLitter','cSoil',
                                          'cBECCS', 'cBiochar', 'fco2fossub'],
                              var_name='Variable', value_name='Value')
    
    ### Set up figure
    f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, 
                                          sharex=True,
                                          gridspec_kw={'hspace': 0.03},
                                          figsize=(5.5,5))
    plt.rcParams.update({'hatch.color': 'k'})
    
    ### Remove white space
    f.subplots_adjust(top=0.94, bottom=0.1, 
                      right=0.99, left=0.13)

    ### Plot boxplots
    ax_top = sns.boxplot(x='Variable', y='Value', 
                         hue='Experiment', data=df_melted, 
                         palette=['tab:grey','#e6a176','#00678a'],
                         showfliers=False,ax=ax_top)
            
    ax_bottom = sns.boxplot(x='Variable', y='Value', 
                            hue='Experiment', data=df_melted, 
                            palette=['tab:grey','#e6a176','#00678a'],
                            showfliers=False,ax=ax_bottom)

    # Get current y-limits
    y_min, y_max = ax_top.get_ylim()

    if plot_type != 'average':
        if region == 'Global':
            ax_top.set_ylim(bottom=120.01)
            ax_bottom.set_ylim(-3,83)
        elif region == 'ACTO':
            ax_top.set_ylim(bottom=20)
            ax_bottom.set_ylim(-3,14)            
        elif region == 'NAMERICA':
            ax_top.set_ylim(bottom=22.51)
            ax_bottom.set_ylim(-4,11)  
        elif region == 'EU27':
            ax_top.set_ylim(bottom=6)
            ax_bottom.set_ylim(-0.5,4.8)
        elif region == 'Boreal':
            ax_top.set_ylim(bottom=38)
            ax_bottom.set_ylim(-1,19.9)
    else:
        if region == 'Global':
            ax_top.set_ylim(bottom=1100)
            ax_bottom.set_ylim(-100,590)
        elif region == 'ACTO':
            ax_top.set_ylim(bottom=1550)
            ax_bottom.set_ylim(-560,950)            
        elif region == 'NAMERICA':
            ax_top.set_ylim(bottom=1130)
            ax_bottom.set_ylim(-100,510)  
        elif region == 'EU27':
            ax_top.set_ylim(bottom=1550)
            ax_bottom.set_ylim(-200,1100)
                    
    ### Remove spines                     
    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)

    ### Remove xticks from top axis
    ax_top.tick_params(axis='x', which='both', 
                       bottom=False, top=False, 
                       labelbottom=False)

    ### Set xticks on the bottom axis
    ax_bottom.set_xticks([0, 1, 2, 3, 4, 5])
    ax_bottom.set_xticklabels(['$\mathit{cVeg}$',
                               '$\mathit{cLitter}$',
                               '$\mathit{cSoil}$',
                               '$\mathit{cBECCS}$',
                               '$\mathit{cBiochar}$',
                               r'$\mathit{Fossil\ fuel}$' + '\n' + r'$\mathit{substitution}$'
                               ])

    ### Remove axis labels
    for axis in (ax_top,ax_bottom):
        axis.set_xlabel('')
        axis.set_ylabel('')

    ### Plot first part
    ax = ax_top
    
    ### Size of diagonal 'break' lines
    d = .01  
    
    ### Plot arguments
    kwargs = dict(transform=ax.transAxes, color='k', linewidth=0.8, clip_on=False)
    
    ### Top-left diagonal
    ax.plot((-d, +d), (-d, +d), **kwargs)        

    ### Plot second part
    ax2 = ax_bottom
    
    ### Switch to the bottom axes
    kwargs.update(transform=ax2.transAxes)  
    
    ### Bottom-left diagonal
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  

    ### Custom label
    if plot_type == 'average':
        label = '$\Delta$ Carbon Pool [gC m$^{-2}$]'
    else:
        label = '$\Delta$ Carbon Pool [PgC]'
    
    if region in ('Global','Temperate'):
        f.text(0.01, 0.4, label, rotation=90, fontsize=10)

    ### Remove one of the legend
    ax_bottom.legend_.remove()
    
    if legend_bool:
        ### Remove frame and title
        ax_top.legend(frameon=False,loc='center right')
        ax_top.legend_.set_title(None)
    else:
        ax_top.legend_.remove()
    
    ax_bottom.axhline(0,color='tab:grey',lw=0.5)
    
    if region == 'Global':
        ax_top.set_title('a)', loc='left')
        ax_top.set_title('Global') 
    elif region == 'Boreal':
        ax_top.set_title('b)', loc='left')
        ax_top.set_title('Boreal')    
    elif region == 'Temperate':
        ax_top.set_title('c)', loc='left')
        ax_top.set_title('Temperate')
    elif region == 'Tropical':
        ax_top.set_title('d)', loc='left')
        ax_top.set_title('Tropical')        
           
    plt.savefig('figures/carbon/'+region+'_'+first_year+'-'+last_year+'_'+plot_type+'.png',dpi=400)

if region == 'Global':
    legend_bool=False
else:
    legend_bool=False

if region in ('Global','Boreal'):
    make_barplot_break(region,legend_bool,first_year,last_year,plot_type)
else:
    make_barplot(region,legend_bool,first_year,last_year,plot_type)
