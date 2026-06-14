import xarray as xr
from xclim.core.calendar import percentile_doy
from xclim.indices import tx90p

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str, required=True)
parser.add_argument('--realisation', type=str, required=True)

args = parser.parse_args()

### Assign variables
exp_id=args.exp_id
realisation=args.realisation

pathwayIN=''
    
def get_tx90p(exp_id,realisation):
    ### Get exp ID and realisation for historical
    if exp_id == 'a7en' and realisation == 'r1i1p1f1':
        realisation='r4i1p1f1'
    
    ### Open DataSet
    da = xr.open_dataset(pathwayIN+exp_id+'_'+realisation+
                            'climate/tasmax/tasmax_day_EC-Earth3-CC_ssp245_'+
                            realisation+'_gr_18500101-21001231.nc').tasmax
    
    ### Get 90th percentile over baseline period over a 5 day moving window
    tasmax_per = percentile_doy(da.sel(time=slice('1981','2010')), 
                                window=5, per=90).sel(percentiles=90)
    
    ### Calculate hot days
    da_tx90p = tx90p(da, tasmax_per)
    
    ### Convert to DataSet
    ds_tx90p = da_tx90p.to_dataset(name='tx90p')
    
    ### Set attributes
    ds_tx90p['lat'].attrs = {'units': 'degrees_north',
                            'standard_name': 'latitude'}
    ds_tx90p['lon'].attrs = {'units': 'degrees_east', 
                            'standard_name': 'longitude'}
    
    ### Save hot days to netcdf
    ds_tx90p.to_netcdf(pathwayIN+exp_id+'_'+realisation+
                       '/climate/tx90p/tx90p_day_EC-Earth3-CC_ssp245_'+
                       realisation+'_gr_1850-2100.nc',
                       encoding = {'time': {'dtype': 'double'},
                                   'lat': {'dtype': 'double'},
                                   'lon': {'dtype': 'double'},
                                   'tx90p': {'dtype': 'float32'}})        
        
get_tx90p(exp_id,realisation)
