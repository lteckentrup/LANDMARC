import xarray as xr
import glob
from xclim import ensembles

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
parser.add_argument('--exp_ref', type=str, required=True)
parser.add_argument('--exp_pert', type=str, required=True)
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)

args = parser.parse_args()

### Assign variables
var=args.var
exp_ref=args.exp_ref
exp_pert=args.exp_pert
first_year=args.first_year
last_year=args.last_year

pathwayIN=''

### Mask ocean points
ds_mask = xr.open_dataset(pathwayIN+'aux/landmask.nc')

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

### Define cmor IDs
ID_to_var_dict = {
    'Lmon': ['cVeg','cLitter'],
    'Emon': ['cLand', 'cSoil'],
    'Eyr': ['cBECCS', 'cBiochar', 'fco2fossub'],
    'Amon': ['tas', 'pr', 'albedo', 'hfls',
             'SPI_1', 'SPI_3', 'SPI_6', 'SPI_12', 
             'SPEI_1', 'SPEI_3', 'SPEI_6', 'SPEI_12'],
    'day': ['FWI', 'FFDI','tx90p','dry_days']
}
var_to_ID_dict = {var: id for id, vars in ID_to_var_dict.items() for var in vars}

### Define fname suffix
suffix_to_var_dict = {
    '_2015-2100.nc': ['cBECCS','cBiochar','fco2fossub','cLand','cSoil','cVeg','cLitter','tx90p'],
    '_185001-210012.nc': ['tas', 'pr', 'albedo','hfls'],
    '_18500101-21001231.nc': ['dry_days']
    }
var_to_suffix_dict = {var: id for id, vars in suffix_to_var_dict.items() for var in vars}

### Find correct directory
### Define fname suffix
directory_to_var_dict = {
    'carbon': ['cBECCS','cBiochar','fco2fossub','cLand','cSoil','cVeg','cLitter'],
    'climate': ['albedo','tas', 'pr', 'hfls','tx90p','dry_days']
    }
var_to_directory_dict = {var: id for id, vars in directory_to_var_dict.items() for var in vars}

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
        
        da_cBECCS['time'] = da_cLand['time']
        da_cBiochar['time'] = da_cLand['time']
        
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

        ### Open dataset
        da = xr.open_dataset(fname)[var]
        
        ### Exclude Antarctica and make sure coordinates match 
        da['lat'], da['lon'] = ds_mask['lat'], ds_mask['lon'] 
                
        ### Cumulative sum for fco2fossub
        if var == 'fco2fossub':
            da_final = da.cumsum(dim='time')

        if var in ('albedo', 'hfls'):
            da_final = time_aggr(da,var)
        else:
            da_final = da
            
        ### Return dataset
        return(da_final.sel(time=slice(first_year,last_year)))
    
def get_ensemble(var,exp,first_year,last_year):
    directory = var_to_directory_dict[var]
    ### Set up files
    exps = [exp] * 3  # Use the same exp for all realizations
    realization_ids = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']

    ### Get datasets
    if var == 'cLand':
        da_r1i1p1f1, da_r2i1p1f1, da_r3i1p1f1 = [
            get_cLand(exp, realization, first_year, last_year)
            for exp, realization in zip(exps, realization_ids)
        ]
    else:           
        da_r1i1p1f1, da_r2i1p1f1, da_r3i1p1f1 = [
            get_data(var, exp, realization, first_year, last_year)
            for exp, realization in zip(exps, realization_ids)
        ]
    
    da = xr.concat([da_r1i1p1f1,
                    da_r2i1p1f1,
                    da_r3i1p1f1],
                   dim='realization')        
    
    ### Return ensemble
    return(da)

def get_significant_change(var,exp_ref,exp_pert,first_year,last_year):
    ### Set up filename
    ID = var_to_ID_dict[var] 
    suffix = var_to_suffix_dict[var] 
    directory = var_to_directory_dict[var]    
    
    ### Get historical reference when both ref and exp_pert are a7eo; 
    ### otherwise define reference and exp_pert 
    if exp_ref == exp_pert:
        first_year_ref,last_year_ref = '1981', '2010'
        if var in ('albedo','hfls','tas','pr'):
            exp_ref = exp_pert
        else:
            exp_ref = 'a766'
    else:
        first_year_ref,last_year_ref = first_year,last_year
        
    ### Get data for experiment
    da_ref = get_ensemble(var,exp_ref,first_year_ref,last_year_ref)
    da_exp = get_ensemble(var,exp_pert,first_year,last_year)

    ### Apply t-test
    fractions = ensembles.robustness_fractions(da_exp,
                                               da_ref,
                                               test='ttest')
  
    ### Return mask where gridpoints are significantly different (threshold 0.05)
    robust_mask = xr.where(fractions.pvals <= 0.05, 1, 0)
        
    fnameOUT = (pathwayIN+'robustness_maps/'+first_year+'-'+last_year+'/'+directory+
                '/'+var+'/'+var+'_'+ID+'_EC-Earth3-CC_ssp245_'+exp_ref+'_'+exp_pert+'_gr.nc')
                                                             
    robust_mask.to_dataset().to_netcdf(fnameOUT)

get_significant_change(var,exp_ref,exp_pert,first_year,last_year)
