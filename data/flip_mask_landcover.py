import numpy as np
import xarray as xr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
parser.add_argument('--biof', type=str, required=True)
parser.add_argument('--exp_ref', type=str, required=True)
parser.add_argument('--exp_pert', type=str, required=True)
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)

args = parser.parse_args()

### Assign variables
var=args.var
biof=args.biof
exp_ref=args.exp_ref
exp_pert=args.exp_pert
first_year=args.first_year
last_year=args.last_year

### Set pathway where input files are located
pathwayIN=

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
        print(fname)
    elif var == 'area_share_afforestation':
        if exp == 'a7cy':
            fname=(pathwayIN+'a7en_r1i1p1f1/area_afforestation_a7en_2015-2100.nc')
            ds = xr.open_dataset(pathwayIN+'a7en_r1i1p1f1/area_afforestation_a7en_2015-2100.nc') 
            ds[var] = ds[var] * 0
            print(fname)
        else:            
            fname=(pathwayIN+exp+'_r1i1p1f1/area_afforestation_'+exp+'_2015-2100.nc')
            ds = xr.open_dataset(pathwayIN+exp+'_r1i1p1f1/area_afforestation_'+
                                 exp+'_2015-2100.nc')
            print(fname)          
    else:
        ### Read in variable
        fname=(pathwayIN+exp+'_r1i1p1f1/lu_frac_'+exp+'_years.nc')
        ds = xr.open_dataset(fname)
        print(fname)

    ### Convert to percentage    
    ds[var] = ds[var] * 100
    
    ### Invert latitudes to match EC-Earth outputs
    ds = ds.reindex(lat=list(reversed(ds['lat'])))
        
    ### Align latitudes and longitudes
    ds['lat'], ds['lon'] = ds_mask['lat'], ds_mask['lon']

    ### Mask ocean points
    da = ds[var].where(ds_mask['sftlf'] != 0, np.nan)
    return(da)

def create_change_mask(var,biof,exp,first_year,last_year):
    ### Get data
    ### End of century value for Reference (a7cy), Moderate CDR Ambition (a7en) or High CDR Ambition (a7eo)
    da = read_in_file(var,biof,exp)
    
    da_ref = da.sel(time=slice('2015','2015')).mean(dim='time')
    da_exp = da.sel(time=slice(first_year,last_year)).mean(dim='time')
    
    ### Mask where both reference periods are 0 everywhere
    da_noC = (da_exp == 0) & (da_ref == 0)

    ### Set points to np.nan where the mask is True
    da_exp_noC = da_exp.where(~da_noC, np.nan)
    da_ref_noC = da_ref.where(~da_noC, np.nan)

    ### Calculate difference between experiment and reference period or experiment
    da = da_exp_noC - da_ref_noC
    
    ### Return binary map where positive changes are 1 and negative -1
    da_binary = xr.where(da > 0, 1, xr.where(da < 0, -1, 0))
          
    return(da_binary)

def create_flip_mask(var,biof,exp_ref,exp_pert,first_year,last_year):
    da_exp_ref = create_change_mask(var,biof,exp_ref,first_year,last_year)
    da_exp_pert = create_change_mask(var,biof,exp_pert,first_year,last_year)
              
    ### When LMTs flip carbon source to carbon sink
    flip_neg_2_pos = (da_exp_ref == -1) & (da_exp_pert == 1)

    ### When LMTs flip carbon sink to carbon source
    flip_pos_2_neg = (da_exp_ref == 1) & (da_exp_pert == -1)

    ### When LMTs don't change sign
    no_change = da_exp_ref == da_exp_pert

    ### Start with all np.nan
    da_mask = xr.full_like(da_exp_ref, fill_value=np.nan, dtype=float)

    ### Apply conditions stepwise
    da_mask = da_mask.where(~flip_pos_2_neg, -1)
    da_mask = da_mask.where(~flip_neg_2_pos, 1)

    ### Save mask to DataSet
    ds_mask = da_mask.to_dataset(name='flip')
    
    ### Attributes
    ds_mask['lat'].attrs = {
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north',
        'axis': 'Y'
    }

    ds_mask['lon'].attrs = {
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east',
        'axis': 'X'
    }
    
    ### Encoding
    encoding = {
        'flip': {'dtype': 'float32'},
        'lat': {'dtype': 'float32'},
        'lon': {'dtype': 'float32'}
    }

    ### Save to netCDF
    suffix = '_sign_flip.nc'        
    fnameOUT = (pathwayIN+'robustness_maps/2071-'+last_year+
                '/landcover/'+var+'/'+var+'_EC-Earth3-CC_ssp245_'+
                exp_ref+'_'+exp_pert+'_gr'+suffix)
    ds_mask.to_netcdf(fnameOUT,encoding=encoding)

create_flip_mask(var,biof,exp_ref,exp_pert,first_year,last_year)  
