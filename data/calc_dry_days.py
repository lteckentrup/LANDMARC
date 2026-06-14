import xarray as xr
from xclim.core.calendar import percentile_doy
from xclim.indices import dry_days

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str, required=True)
parser.add_argument('--realisation', type=str, required=True)

args = parser.parse_args()

### Assign variables
exp_id=args.exp_id
realisation=args.realisation

pathwayIN=''
    
def get_dry_days(exp_id,realisation):
    ### We had to rerun a7en r2i1p1f1 and the new according realisation is r4i1p1f1
    if exp_id == 'a7en' and realisation == 'r2i1p1f1':
        realisation = 'r4i1p1f1'
        
    ### Open projection file
    da = xr.open_dataset(pathwayIN+exp_id+'_'+realisation+
                         'climate/pr/pr_day_EC-Earth3-CC_ssp245_'+
                         realisation+'_gr_18500101-21001231.nc').pr
    
    ### Calculate dry days
    da_dry_days = dry_days(da, thresh='1 mm/d')
    
    ### Convert to DataSet
    ds_dry_days = da_dry_days.to_dataset(name='dry_days')
    
    ### Set attributes
    ds_dry_days['lat'].attrs = {'units': 'degrees_north',
                                'standard_name': 'latitude'}
    ds_dry_days['lon'].attrs = {'units': 'degrees_east',
                                'standard_name': 'longitude'}
    
    ### Save hot days to netcdf
    ds_dry_days.to_netcdf(pathwayIN+exp_id+'_'+realisation+
                          '/climate/dry_days/dry_days_day_EC-Earth3-CC_ssp245_'+
                          realisation+'_gr_1850-2100.nc',
                          encoding = {'time': {'dtype': 'double'},
                                      'lat': {'dtype': 'double'},
                                      'lon': {'dtype': 'double'},
                                      'dry_days': {'dtype': 'float32'}})

get_dry_days(exp_id,realisation)
