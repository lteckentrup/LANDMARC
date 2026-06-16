import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt

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
    'Amon': ['tas', 'pr','hfls', 'albedo'],
    'day': ['tx90p','dry_days','co2mass','co2s']
}
var_to_ID_dict = {var: id for id, vars in ID_to_var_dict.items() for var in vars}

### Find correct directory
directory_to_var_dict = {
    'carbon': ['cBECCS','cBiochar','fco2fossub','cLand','cSoil','cVeg','cLitter','co2s','co2mass'],
    'climate': ['albedo','tas', 'pr', 'PET','hfls', 'tx90p','dry_days'],
    'drought': ['SPI_1', 'SPI_3', 'SPI_6', 'SPI_12',
                'SPEI_1', 'SPEI_3', 'SPEI_6', 'SPEI_12'],
    'fire': ['FFDI', 'FWI']
    }
var_to_directory_dict = {var: id for id, vars in directory_to_var_dict.items() for var in vars}
 
### Define area weighted sum / average
def area_weighted_stats(da, var, mask_type):
    ### Align coordinates
    for ds in (ds_mask, ds_gridarea):
        ds['lat'], ds['lon'] = da['lat'], da['lon']

    ### Mask land or locean
    if mask_type == 'land_only':
        da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf']))
        da = da.where(~np.isnan(ds_mask['sftlf']))
    elif mask_type == 'ocean_only':
        da_gridarea = ds_gridarea['cell_area'].where(np.isnan(ds_mask['sftlf']))
        da = da.where(np.isnan(ds_mask['sftlf']))
    elif mask_type == 'Global':
        da_gridarea = ds_gridarea['cell_area']
        da = da  

    ### Mask regions
    if mask_type == 'Boreal':
        da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf'])).sel(lat=slice(50, 90))
        da = da.where(~np.isnan(ds_mask['sftlf'])).sel(lat=slice(50, 90))

    elif mask_type == 'Temperate':
        latmask = (((ds_gridarea['cell_area'].lat >= -50) & (ds_gridarea['cell_area'].lat <= -30)) |
                   ((ds_gridarea['cell_area'].lat >= 30) & (ds_gridarea['cell_area'].lat <= 50)))
        da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf'])).where(latmask, drop=True)
        da = da.where(~np.isnan(ds_mask['sftlf'])).where(latmask, drop=True)

    elif mask_type == 'Tropical':
        da_gridarea = ds_gridarea['cell_area'].where(~np.isnan(ds_mask['sftlf'])).sel(lat=slice(-30, 30))
        da = da.where(~np.isnan(ds_mask['sftlf'])).sel(lat=slice(-30, 30))

    ### Multiply with area weights
    da_weights = da * da_gridarea

    ### Get sum for carbon pools and averages for all others
    if var in ('cLand', 'cBECCS', 'cBiochar', 'fco2fossub'):
        da_weighted = da_weights.sum(dim=['lat', 'lon']) / 1e12
    else:
        da_weighted = da_weights.sum(dim=['lat', 'lon']) / da_gridarea.sum(dim=['lat', 'lon'])

    ### Return array
    return(da_weighted.values.flatten())

### Get annual values
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

### Get cLand
def get_cLand(exp,realisation):
    ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
    if exp == 'a7en' and realisation == 'r2i1p1f1':
        realisation = 'r4i1p1f1'
    
    ### For historical, cLand = cLandAll, for projections it's the sum of cLand, BECCS and Biochar  
    if exp == 'a766':
        da_cLandAll = get_data('cLand',exp,realisation)
    else:
        #### Read in separate carbon pools
        da_cLand = get_data('cLand',exp,realisation)
        da_cBECCS = get_data('cBECCS',exp,realisation)
        da_cBiochar = get_data('cBiochar',exp,realisation)
        da_fco2fossub = get_data('fco2fossub',exp,realisation)
        
        ### Align time axes
        da_cBECCS['time'] = da_cLand['time']
        da_cBiochar['time'] = da_cLand['time']
        da_fco2fossub['time'] = da_cLand['time']
        
        ### Sum to get all carbon stored on land
        da_cLandAll = da_cLand + da_cBECCS + da_cBiochar + da_fco2fossub.cumsum(dim='time')

    return(da_cLandAll)

### Read in data
def get_data(var,exp,realisation): 
    ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
    if exp == 'a7en' and realisation == 'r2i1p1f1':
        realisation = 'r4i1p1f1'
        
    ### Get CMIP ID
    ID = var_to_ID_dict[var]
    directory = var_to_directory_dict[var]
    
    ### Define grid
    if var in ('co2s','co2mass'):
        grid = 'gn'
    else:
        grid = 'gr'
        
    ### Climate files are merged to 1850-2100, carbon files are split into historical and projection
    if exp  == 'a766':
        first_year, last_year = '1981', '2010'
        CMIP_scen, suffix, realisation = 'historical', '1850-2014.nc', 'r1i1p1f1'
    else:
        if directory == 'carbon':
            if var in ('co2s', 'co2mass'):
                CMIP_scen, suffix = 'ssp245', '20150101-21001231.nc'
            else:
                CMIP_scen, suffix = 'ssp245', '2015-2100.nc'
        else:
            if var in ('tx90p', 'dry_days'):
                CMIP_scen, suffix = 'ssp245', '1850-2100.nc'
            else:
                CMIP_scen, suffix = 'ssp245', '185001-210012.nc'
    
    ### cBECCS, cBiochar and fco2fossub are 0 in historical - no need to read in
    if not (var in ('cBECCS', 'cBiochar', 'fco2fossub') and exp == 'a766'):
        ### Create file name
        fname=(pathwayIN+exp+'_'+realisation+'/'+directory+'/'+
               var+'/'+var+'_'+ID+'_EC-Earth3-CC_'+CMIP_scen+
               '_'+realisation+'_'+grid+'_'+suffix)
        print(fname)
        ### Open dataset
        da = xr.open_dataset(fname)[var]
        
        ### Make sure coordinates match 
        if var in ('co2s','co2mass'):
            pass
        else:
            da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon'] 
        
        ### Get annual averages
        if var in ('albedo', 'hfls'):
            da_final = time_aggr(da,var)
        else:
            da_final = da
                
        ### Return dataset
        return(da_final)

def get_dataframe_input(var,exp,realisation,mask_type):
    if var == 'cLand':
        da_raw = get_cLand(exp,realisation)
    else:
       da_raw = get_data(var,exp,realisation)
    
    if var in ('tas','pr'):
        da = time_aggr(da_raw,var)
    else:
        da = da_raw
    
    if var in ('co2s','co2mass'):
        np_regional = da_raw.values.flatten()
    else:
        np_regional = area_weighted_stats(da, var, mask_type)
    
    if var in ('tas', 'pr'):
        if exp == 'a7en' and realisation == 'r2i1p1f1':
            np_regional[-14]=np.nan
        
    return(np_regional)

def get_mitigation_potential(exp_pert,realisation):
    np_ref_cLand = get_dataframe_input('cLand','a7cy',realisation,'land_only') 
    np_ref_co2mass = get_dataframe_input('co2mass','a7cy',realisation,'')
    
    np_exp_cLand = get_dataframe_input('cLand',exp_pert,realisation,'land_only')
    np_exp_co2mass = get_dataframe_input('co2mass',exp_pert,realisation,'')
    
    np_diff_co2mass = -1*(np_exp_co2mass - np_ref_co2mass)
    np_diff_cLand = np_exp_cLand - np_ref_cLand
    
    np_rel = 100*(np_diff_co2mass/np_diff_cLand)
    
    return(np_rel)

def get_dataframe(var,exp_id,mask_type,first_year,last_year,smooth):
    df = pd.DataFrame()
    if var == 'co2_growth_rate':       
        df['r1i1p1f1'] = get_dataframe_input('co2s',exp_id,'r1i1p1f1',mask_type)
        df['r2i1p1f1'] = get_dataframe_input('co2s',exp_id,'r2i1p1f1',mask_type)
        df['r3i1p1f1'] = get_dataframe_input('co2s',exp_id,'r3i1p1f1',mask_type)
        df['time'] = np.arange(2101 - len(df),2101,1)
        df.set_index('time',inplace=True)
        df_diff = df.diff()
    elif var == 'mitigation_potential':
        df['r1i1p1f1'] = get_mitigation_potential(exp_id,'r1i1p1f1')
        df['r2i1p1f1'] = get_mitigation_potential(exp_id,'r2i1p1f1')
        df['r3i1p1f1'] = get_mitigation_potential(exp_id,'r3i1p1f1')
        df['time'] = np.arange(2101 - len(df),2101,1)
        df.set_index('time',inplace=True)
        df_diff = df
    else:
        df['r1i1p1f1'] = get_dataframe_input(var,exp_id,'r1i1p1f1',mask_type)
        df['r2i1p1f1'] = get_dataframe_input(var,exp_id,'r2i1p1f1',mask_type)
        df['r3i1p1f1'] = get_dataframe_input(var,exp_id,'r3i1p1f1',mask_type)
        df['time'] = np.arange(2101 - len(df),2101,1)
        df.set_index('time',inplace=True)
        if var in ('co2s', 'co2mass'):
            df_diff = df
        else:
            df_diff = df.loc[2015:2100] - df.loc[first_year:last_year].mean()
     
    if smooth == False:
        pass
    else:
        df_diff = df_diff.rolling(window=smooth,center=True,min_periods=1).mean()
        
    df_ens = pd.DataFrame()
    df_ens['ensmean'] = df_diff.mean(axis=1)
    df_ens['min'] = df_diff.min(axis=1)
    df_ens['max'] = df_diff.max(axis=1)
    
    ### Include identifier for exp and region
    if exp_id == 'a7cy':
        df_ens['Experiment'] = 'Reference'
    elif exp_id == 'a7en':
        df_ens['Experiment'] = 'Moderate CDR Ambition'
    elif exp_id == 'a7eo':
        df_ens['Experiment'] = 'High CDR Ambition'
    
    if mask_type == 'ocean_only':
        df_ens['Region'] = 'Ocean'
    elif mask_type == 'land_only':
        df_ens['Region'] = 'Land' 
    else:
        df_ens['Region'] = mask_type
        
    if var == 'mitigation_potential':
        return(df_ens.loc[2035:2100])
    else:
        return(df_ens)
            
### Define function to find first time where majority diverges
def time_of_departure(df_outside, years, threshold):
    itemindex = np.where((df_outside == 1) | (df_outside == -1))
    year_ToD = np.nan
    
    for i in itemindex[0]:
        proportion = np.absolute(df_outside[i:].sum() / len(df_outside[i:]))
        if proportion >= threshold:
            year_ToD = years[i]
            break
    
    if year_ToD >= 2096 or year_ToD <= 2020:
        return(np.nan)
    else:
        if np.isnan(year_ToD):
            return(np.nan)
        else:
            return(year_ToD)
        
def time_of_reversal(data, scenario, threshold):
    ### Boolean mask to test where negative
    df = pd.DataFrame()
    df[scenario] = data
    df['time'] = np.arange(2101-len(df),2101,1)
    df.set_index('time',inplace=True)
    
    df_reversed = df[scenario].dropna() <= 0
    itemindex = np.where(df_reversed == 1)
    years = df_reversed.index
    
    year_reversal = np.nan
    
    for i in itemindex[0]:
        proportion = np.absolute(df_reversed[i:].sum() / len(df_reversed[i:]))
        if proportion >= threshold:
            year_reversal = years[i]
            break
    
    if year_reversal >= 2096 or year_reversal <= 2020:
        return(np.nan)
    else:
        if np.isnan(year_reversal):
            return(np.nan)
        else:
            return(year_reversal)
     
def plot_timeseries(axis,var,mask_type,smooth):
    ### Get reference timeperiod
    if var in ('tas','pr'):
        first_year, last_year = 1850, 1879
    else:
        first_year, last_year = 1981, 2010
    
    ### Get DataFrame  
    df_a7cy = get_dataframe(var,'a7cy',mask_type,first_year,last_year,smooth)
    df_a7en = get_dataframe(var,'a7en',mask_type,first_year,last_year,smooth)
    df_a7eo = get_dataframe(var,'a7eo',mask_type,first_year,last_year,smooth)
    
    ### Select projection
    df_a7cy_sel = df_a7cy.loc[2015:2100]
    df_a7en_sel = df_a7en.loc[2015:2100]
    df_a7eo_sel = df_a7eo.loc[2015:2100]
    
    ### Plot ensemble mean
    axis.plot(df_a7cy_sel.index,df_a7cy_sel['ensmean'],color='k',label='Reference')
    axis.plot(df_a7en_sel.index,df_a7en_sel['ensmean'],color='#e6a176',label='Moderate CDR Ambition')
    axis.plot(df_a7eo_sel.index,df_a7eo_sel['ensmean'],color='#00678a',label='High CDR Ambition')
    
    ### Plot ensemble spread
    axis.fill_between(df_a7cy_sel.index,df_a7cy_sel['min'],df_a7cy_sel['max'],color='k',alpha=0.4)
    axis.fill_between(df_a7en_sel.index,df_a7en_sel['min'],df_a7en_sel['max'],color='#e6a176',alpha=0.4)
    axis.fill_between(df_a7eo_sel.index,df_a7eo_sel['min'],df_a7eo_sel['max'],color='#00678a',alpha=0.4)
    
    ### Include ScenarioMIP CO2 concentration for SSP1-1.9 and SSP1-2.6
    if var in ('co2s','co2_growth_rate'):        
        ### Get DataArrays
        da_ref_ssp119 = xr.open_dataset(pathwayIN+'ssp119/co2obs_ssp119.nc').co2s
        da_ref_ssp126 = xr.open_dataset(pathwayIN+'ssp126/co2obs_ssp126.nc').co2s
        
        ### Convert to numpy
        np_ref_ssp119 = da_ref_ssp119.values.flatten()[:-1]
        np_ref_ssp126 = da_ref_ssp126.values.flatten()[:-1]
        
        if var == 'co2_growth_rate':
            da_ref_ssp119 = da_ref_ssp119.diff(dim='time')
            da_ref_ssp126 = da_ref_ssp126.diff(dim='time')
            np_ref_ssp119 = np.insert(da_ref_ssp119.values.flatten()[:-1],0,np.nan)
            np_ref_ssp126 = np.insert(da_ref_ssp126.values.flatten()[:-1],0,np.nan)
        
        ### Plot ScenarioMIP for reference   
        axis.plot(df_a7en_sel.index,np_ref_ssp126,color='#e6a176',ls=':',label='SSP1-2.6')
        axis.plot(df_a7eo_sel.index,np_ref_ssp119,color='#00678a',ls=':',label='SSP1-1.9')
        
    ## Add ToD and ToR
    if var in ('tas','pr','co2s'):
        ### Find ToD for LMTs vs Reference        
        ### Drop missing year from all runs
        df_a7cy.drop(2087,inplace=True)
        df_a7en.drop(2087,inplace=True)
        df_a7eo.drop(2087,inplace=True)

        ### Set up DataFrames to test ToD/ToR
        ### Moderate CDR Ambition
        df_outside_MOD = pd.Series(0, index=df_a7cy['ensmean'].index)
        df_outside_MOD[df_a7en['ensmean'] < df_a7cy['min']] = -1
        df_outside_MOD[df_a7en['ensmean'] > df_a7cy['max']] = 1
        
        ### High CDR Ambition
        df_outside_HIGH = pd.Series(0, index=df_a7cy['ensmean'].index)
        df_outside_HIGH[df_a7eo['ensmean'] < df_a7cy['min']] = -1
        df_outside_HIGH[df_a7eo['ensmean'] > df_a7cy['max']] = 1
        
        ### Get years
        years = df_a7cy.index
        
        ### Get time of departure from reference
        ToD_MOD = time_of_departure(df_outside_MOD, years, 0.8)
        ToD_HIGH = time_of_departure(df_outside_HIGH, years, 0.8)
        
        ### Plot ToD
        ### Get minimum value for vlines
        y_min, y_max = axis.get_ylim()
    
        if ToD_MOD == ToD_HIGH:
            ls_MOD='--'
            ls_HIGH='-'
        else:
            ls_MOD='-'
            ls_HIGH='-'
            
        ### Plot vlines ending at timeseries line; overlay if ToD is identical
        if np.isnan(ToD_MOD):
            pass
        else:
            axis.vlines(ToD_MOD,
                        ymin=0,
                        ymax=df_a7en.loc[ToD_MOD]['ensmean'],
                        color = '#e6a176',ls=ls_MOD,lw=3)
        if np.isnan(ToD_HIGH):
            pass
        else:
            axis.vlines(ToD_HIGH,
                        ymin=0,
                        ymax=df_a7eo.loc[ToD_HIGH]['ensmean'],
                        color = '#00678a',ls=ls_HIGH,lw=3)

        axis.set_ylim(0.97*y_min,1.03*y_max)
        
        ### Annotate years
        if ToD_MOD == ToD_HIGH:            
            ### Annotate year
            axis.annotate(str(ToD_MOD),
                          xy=(ToD_MOD, 0),
                          xycoords=('data', 'axes fraction'),
                          xytext=(0, 5),
                          textcoords='offset points',
                          ha='left', va='bottom')
        else:            
            ### Annotate years
            axis.annotate(str(ToD_MOD),
                          xy=(ToD_MOD, 0),
                          xycoords=('data', 'axes fraction'),
                          xytext=(30, 5),
                          textcoords='offset points',
                          ha='right', va='bottom',color='#e6a176')
            
            axis.annotate(str(ToD_HIGH),
                          xy=(ToD_HIGH, 0),
                          xycoords=('data', 'axes fraction'),
                          xytext=(-30, 5),
                          textcoords='offset points',
                          ha='left', va='bottom',color='#00678a')

    if var == 'co2_growth_rate':
        ### Get Time of Reversal
        ToR_ssp119 = time_of_reversal(np_ref_ssp119, 'SSP1-1.9', 0.8)
        ToR_ssp126 = time_of_reversal(np_ref_ssp126, 'SSP1-2.6', 0.8)
        ToR_MOD = time_of_reversal(df_a7en_sel['ensmean'].values.flatten(), 'Moderate CDR Ambition', 0.8)
        ToR_HIGH = time_of_reversal(df_a7eo_sel['ensmean'].values.flatten(), 'High CDR Ambition', 0.8)
                
        ### Plot ToR
        ### Get minimum value for vlines
        y_min, y_max = axis.get_ylim()
        
        ### Plot vlines ending at timeseries line
        axis.vlines(ToR_ssp119,
                    ymin=-10,
                    ymax=np_ref_ssp119[ToR_ssp119-2015],
                    color = '#00678a',ls=':')
        axis.vlines(ToR_ssp126,
                    ymin=-10,
                    ymax=np_ref_ssp126[ToR_ssp126-2015],
                    color = '#e6a176',ls=':')
        axis.vlines(ToR_MOD,
                    ymin=-10,
                    ymax=df_a7en['ensmean'].loc[ToR_MOD],
                    color = '#e6a176')
        axis.vlines(ToR_HIGH,
                    ymin=-10,
                    ymax=df_a7eo['ensmean'].loc[ToR_HIGH],
                    color = '#00678a')
                
        ### Annotate years
        axis.annotate(str(ToR_MOD),
                      xy=(ToR_MOD, 0),
                      xycoords=('data', 'axes fraction'),
                      xytext=(5, 5),
                      textcoords='offset points',
                      ha='left', va='bottom',color='#e6a176')
        
        axis.annotate(str(ToR_HIGH),
                      xy=(ToR_HIGH, 0),
                      xycoords=('data', 'axes fraction'),
                      xytext=(-5, 5),
                      textcoords='offset points',
                      ha='right', va='bottom',color='#00678a')
        
        axis.annotate(str(ToR_ssp119),
                      xy=(ToR_ssp119, 0),
                      xycoords=('data', 'axes fraction'),
                      xytext=(-5, 5),
                      textcoords='offset points',
                      ha='right', va='bottom',color='#00678a')
        
        axis.annotate(str(ToR_ssp126),
                      xy=(ToR_ssp126, 0),
                      xycoords=('data', 'axes fraction'),
                      xytext=(-5, 5),
                      textcoords='offset points',
                      ha='right', va='bottom',color='#e6a176')
        
        axis.set_ylim(1.08*y_min,1.03*y_max)
                        
    ### Remove spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    ### Drop xlabels
    axis.set_xlabel('')
    
    ### Set x limit
    axis.set_xlim(2010,2105)

def plot_boxplot(axis,var,smooth):    
    ### Get reference timeperiod
    if var in ('tas','pr'):
        first_year, last_year = 1850, 1879
    else:
        first_year, last_year = 1981, 2010
    
    exp_ids = ['a7cy', 'a7en', 'a7eo']
    if var in ('tas','pr'):
        ### Get ensemble dataframe
        regions = ['Global', 'ocean_only', 'land_only']

    else:
        ### Get ensemble dataframe
        regions = ['land_only', 'Boreal', 'Temperate', 'Tropical']

    ### Initialise list for DataFrames
    dfs = []

    ### Loop through experiments and regions and append DataFrames
    for exp in exp_ids:
        for region in regions:
            dfs.append(get_dataframe(var, exp, region, first_year, last_year, smooth))

    ### Concat all DataFrames
    df = pd.concat(dfs)
                    
    ### Select last 30 years
    df = df[(df.index >= 2071) & (df.index <= 2100)]
    df_ensmean = df[['ensmean','Region','Experiment']]
    
    ### Plot boxplots
    axis = sns.boxplot(data=df_ensmean,
                       x='Region',
                       y='ensmean',
                       hue='Experiment',
                       showfliers=False,
                       palette=['tab:grey','#e6a176','#00678a'],
                       ax=axis)

    ### Remove spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    ### Drop xlabels
    axis.set_xlabel('')
    axis.set_ylabel('')
    
    ### Remove legend
    axis.legend_.remove()
                
### Set up matplotlib figure with subplots
fig=plt.figure(figsize=(12,12))
ax1=fig.add_subplot(3,3,1)
ax2=fig.add_subplot(3,3,2)
ax3=fig.add_subplot(3,3,3)  
ax4=fig.add_subplot(3,3,4)
ax5=fig.add_subplot(3,3,5)
ax6=fig.add_subplot(3,3,6)  
ax7=fig.add_subplot(3,3,7)
ax8=fig.add_subplot(3,3,8)
ax9=fig.add_subplot(3,3,9)      

### Plot timeseries
plot_timeseries(ax1,'mitigation_potential','',20)
plot_timeseries(ax2,'co2s','',False)
plot_timeseries(ax3,'co2_growth_rate','',10)
plot_timeseries(ax4,'tas','Global',10)
plot_boxplot(ax5,'tas',False)
plot_boxplot(ax6,'tx90p',False)
plot_timeseries(ax7,'pr','Global',10)
plot_boxplot(ax8,'pr',False)
plot_boxplot(ax9,'dry_days',False)


### Set ylabels
ax1.set_ylabel('Mitigation efficiency [%]')
ax2.set_ylabel('Atmospheric CO$_2$ [ppm]')
ax3.set_ylabel('Atmospheric CO$_2$ Growth Rate [ppm]')
ax4.set_ylabel('$\Delta$ Temperature [K]')
ax5.set_ylabel('$\Delta$ Temperature [K]')
ax6.set_ylabel('$\Delta$ TX90p [# days]')
ax7.set_ylabel('$\Delta$ Precipitation [mm]')
ax8.set_ylabel('$\Delta$ Precipitation [mm]')
ax9.set_ylabel('$\Delta$ Dry days [# days]')

### Set titles
ax1.set_title('a)',loc='left')
ax2.set_title('b)',loc='left')
ax3.set_title('c)',loc='left')
ax4.set_title('d)',loc='left')
ax5.set_title('e)',loc='left')
ax6.set_title('f)',loc='left')
ax7.set_title('g)',loc='left')
ax8.set_title('h)',loc='left')
ax9.set_title('i)',loc='left')

### Set titles
ax1.set_title('Mitigation efficiency')
ax2.set_title('Atmospheric CO$_2$ concentration')
ax3.set_title('Atmospheric CO$_2$ Growth Rate')
ax4.set_title('$\Delta$ Temperature')
ax5.set_title('$\Delta$ Temperature')
ax6.set_title('$\Delta$ TX90p')
ax7.set_title('$\Delta$ Precipitation')
ax8.set_title('$\Delta$ Precipitation')
ax9.set_title('$\Delta$ Dry days')

### Plot 1.5 and 2 degree target for context in tas     
ax4.axhline(1.5,color='#c0affb',lw=2,zorder=0)
ax4.axhline(2,color='#984464',lw=2,zorder=0)

# ax4.set_ylim(1.1,4.9)
# ax5.set_ylim(1.1,4.9)

# ax7.set_ylim(15,110)
# ax8.set_ylim(15,110)

### Remove frame from legend
ax2.legend(loc='best',
           frameon=False,
           fontsize=9)

### Plot horizontal line
ax3.axhline(0,color='tab:grey',lw=0.5,zorder=0) 
ax9.axhline(0,color='tab:grey',lw=0.5,zorder=0) 
    
### Align y-labels
fig.align_ylabels()

### Tight layout
plt.tight_layout()

### Save figure
plt.savefig('figures/tCDR_impacts.png',dpi=400)
