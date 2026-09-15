
# quick subroutine to analyze the W data
import sys
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
#import netCDF4 as nc
import os
import glob
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
from datetime import datetime, timedelta
#from scipy import interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
import json
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#M = np.array([3, 5e-5, 0.5, 1.5, 5e-4, 0.5, 300e-6, 1000, 1., 800, 50, 1, 1])
m= [49.5, 27.9, 307.2, 0.5, 0.06, 0.02, 10., 50., 223.] 
s = [37.1, 15.9, 16.1, 1, 0.06, 0.02, 5., 20., 10.]
np.set_printoptions(precision=3)
#['swcf', 'lwcf', 'T2M', 'CLDTT', 'LWP', 'IWP', 'MDSCLDSZWTR',  'MDSCLDSZICE', 'TROPT']


def set_atts(dsx, var="Wstd"):
    ds =  dsx.copy()
    fill_value=1.e+15
    # coordinate attributes
    ds["lon"].attrs['long_name'] = 'longitude'
    ds["lon"].attrs['standard_name']     = 'longitude'
    ds["lon"].attrs['units']             = 'degrees_east'
    #ds["lon"].attrs['_FillValue']        = np.array(fill_value, np.float32) 

    ds["lat"].attrs['long_name']         = 'latitude'
    ds["lat"].attrs['standard_name']     = 'latitude'
    ds["lat"].attrs['units']             = 'degrees_north'
    #ds["lat"].attrs['_FillValue']        = np.array(fill_value, np.float32)

    date = ds['time'].values[0] #datetime(2005, 1, 1, 0, 0, 0)
    date1 = ds['time'].values[1]   
    date  =  pd.to_datetime(date)
    date1  =  pd.to_datetime(date1)
    
    #print(date, date1)
    td          = date1-date
    time_increment          = int('{hours}{minutes}{seconds}'.format(hours=int(td.seconds/3600), minutes=int(td.seconds/60), seconds=int(td.seconds)))  
    begin_date              = int(date.strftime('%Y%m%d'))
    begin_time              = int(date.strftime('%H%M%S'))
    #time_increment =  np.array(6000, dtype=np.float32)

    #ds["time"].attrs['long_name'] = 'time'
    ds["time"].attrs['time_increment'] =np.array(time_increment, dtype=np.int32)
    #ds["time"].attrs['units']            = 'days since {:%Y-%m-%d %H:%M:%S}'.format(date)
    ds["time"].attrs['begin_date']  = np.array(begin_date,     dtype=np.int32)
    ds["time"].attrs['begin_time']  = np.array(begin_time,     dtype=np.int32)

    ds[var].attrs["contact"] = "Donifan Barahona, donifan.o.barahona@nasa.gov" 
    ds[var].attrs["fmissing_value"] = np.array(fill_value, np.float32)
    ds[var].attrs["missing_value"] = np.array(fill_value, np.float32)   
    ds[var].attrs["vmin"]            = np.array(-fill_value, np.float32)
    ds[var].attrs["vmax"]            = np.array(fill_value, np.float32)
    ds[var].attrs["valid_range"]     = np.array((-fill_value, fill_value), np.float32) 

    return ds

def nb(x, prec=3):
    	return np.format_float_scientific(x, precision=prec)
 
def amean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    #weights = np.cos(0.01745329*ds.lat)
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(("lon", "lat"),  skipna=True)
    
def corrcoefxy( x, y ):    
    mean_x = np.mean( x )
    mean_y = np.mean( y )
    std_x  = np.std ( x )
    std_y  = np.std ( y )
    n      = len    ( x )
    return (( x - mean_x ) * ( y - mean_y )).sum() / n / ( std_x * std_y )
    
def standardize(ds, m, s):
  i = 0
  for v in  ds.data_vars:
   ds[v] = (ds[v] - m[i])/s[i]
   i = i+1   
  return ds

#def power_scaler(ds, nexp):
#   func  = lambda x, n: np.power(np.where(x<0, 0, x), n)  
#   return xr.apply_ufunc(func, ds, nexp, dask='parallelized') 

def linear_scaler(ds, cst):
   func  = lambda x, n: x*n 
   return xr.apply_ufunc(func, ds, cst, dask='parallelized') 
 



def save_stats(var = 'MOD_CDNC_T258', pth=''):

    chk  = {"time": 1}
    X  =  xr.open_mfdataset(pth, parallel=True, chunks=chk)[[var]]
    print(X)
    Xmean  =  X.mean('time').rename({var:var+'_mean'})
    Xstd  =  X.std('time').rename({var:var+'_std'})  
    Xstats =  xr.merge([Xmean, Xstd])  
    print('-----stats------', Xstats)
    Xstats.to_netcdf(var+ '_stats.nc4', mode = "w")
    return Xstats
    


if __name__ == '__main__':
   #client = Client()
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Check for multiple GPUs

    prefix= '200'

   # v  =  'MOD_CDNC_T268'
   # pth= f"../data/MODIS_ND_DAILY/GiOCEAN_e1.mod_inst_1D_glo_L720x361_sfc_ND.{prefix}*.nc4"  
   # Y = save_stats(pth = pth, var =  v) 
    
   # v  =  'MOD_CDNC_T63'
   # pth= f"../data/MODIS_ND_DAILY/GiOCEAN_e1.mod_inst_1D_glo_L720x361_sfc_ND.{prefix}*.nc4"  
   # Y = save_stats(pth = pth, var =  v) 
   
   pth =  f'{pth_all}dailies/{vars_out[v]}/{prefix}'
   pth_all = '/gpfsm/dnb07/projects/p281/GiOcean_data/'
   
   vars_out = {
        'NCPL_VOL': 'aci_tavg_1dy_glo_L360x181_p27'
    }

    v =  'NCPL_CLDBASE'
    pth= f"../data/NDvar/extracted_2D_GiOCEAN_e1.aci_tavg_1dy_glo_L720x361_sfc.daily.{prefix}*.nc4"
    print(pth) 
    Y = save_stats(pth = pth, var =  v)

  

