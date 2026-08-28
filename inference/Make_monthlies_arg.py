import numpy as np
import sys
#import matplotlib

#import matplotlib.pyplot as plt
#import pandas as pd
#import netCDF4 as nc
import os
import glob
import xarray as xr
import time
from dask.distributed import Client, LocalCluster
import datetime
from datetime import datetime, timedelta
import pandas as pd
import gc


#for now we base all anomaly correlations on MERRA
#new_p =  [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 
#    600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40, 30, 20,
#    10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02]


new_p = np.arange(25, 1005, 25)
new_p = [0.1, 1, 5, 10, new_p]   
#new_p= [1000, 985, 970, 955, 940, 925, 910, 895, 880, 865, 850, 835, 820, 805, 790, 775, 760,
#        745, 730, 715, 700, 685, 670, 655, 640, 625, 610, 595, 580, 565, 550, 535, 520, 505, 490, 475, 
#        460, 445, 430, 415, 400, 385, 370, 355, 340, 325, 310, 295, 280, 265, 250, 235, 220, 205, 190, 175, 
#        160, 145, 130, 115, 100, 85, 70, 55, 40, 30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 
#        0.2, 0.1, 0.07, 0.05, 0.04, 0.03, 0.02]
#new_p =  new_p[::-1]
fill_value=1.e+15

def set_atts_all_vars(dsx, fill_value=1.e15):
    ds = dsx.copy()

    # -------------------------
    # Coordinate attributes
    # -------------------------
    if "lon" in ds:
        ds["lon"].attrs.update({
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        })

    if "lat" in ds:
        ds["lat"].attrs.update({
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        })

    if "lev" in ds:
        ds["lev"].attrs.update({
            "long_name": "pressure",
            "standard_name": "air_pressure",
            "units": "hPa",
            "positive": "down",
            "axis": "Z",
        })

    if "mode" in ds:
        ds["mode"].attrs.update({
            "long_name": "EOF mode index",
            "axis": "E",
        })

    # -------------------------
    # Time attributes (GEOS-style)
    # -------------------------
    if "time" in ds:
        date0 = pd.to_datetime(ds["time"].values[0])

        ds["time"].attrs.update({
            "begin_date": np.int32(date0.strftime("%Y%m%d")),
            "begin_time": np.int32(date0.strftime("%H%M%S")),
            "time_increment": np.int32(732000),  # monthly (GEOS convention)
        })

    # -------------------------
    # Variable attributes (ALL vars)
    # -------------------------
    for v in ds.data_vars:
        ds[v].attrs.setdefault("long_name", v)
        ds[v].attrs.setdefault("standard_name", v)
        ds[v].attrs.setdefault("units", "1")

        ds[v].attrs.update({
            "missing_value": np.float32(fill_value),
            "fmissing_value": np.float32(fill_value),
            "valid_range": np.array(
                (-fill_value, fill_value), dtype=np.float32
            ),
            "vmin": np.float32(-fill_value),
            "vmax": np.float32(fill_value),
            "contact": "Donifan Barahona, donifan.o.barahona@nasa.gov",
        })

    return ds


 
        
if __name__ == '__main__':
   
    fill_value=1.e+15
    #client = Client()
    
    
    yr_st = 2003
    yr_end=  2025
    
    #yr  =  sys.argv[1]
    #yr_st  =  int(yr)
    #yr_end = yr_st+ 1
    
    yrs =range(yr_st, yr_end)
    mos = range(1, 13)
    #mos =  [12]

    dirx =  "./NDnet_infer_daily/"
    for yr in yrs:
    	
        for mo in mos:
            y =  str(yr)
            m =  str(mo).zfill(2)
            #        d =  str(t.day).zfill(2)

            tit =  f"./monthlies/MODIS.tavgM3_3d_wstd_NDinc.{y}{m}.nc4"
            if os.path.exists(tit): #skip if already calculated (Srun is re-starting the code for some reason)
                print(tit + '_____already computed')
                continue

            fil = f"{dirx}NDnet_infer_{y}{m}*.nc4"
            print('processing--------', fil)             
            try:
            	W = xr.open_mfdataset(fil, parallel=True)
            except:
                W = xr.open_mfdataset(fil, parallel=True)
                print('Error______', y+m)
                continue
            W =  W.where(W > -900 )
            W =  W.where(W != fill_value)
            W =  W.where(W < 1e10) 
            #print(dat1)  
            
            print(W)
               
            
              
            ##calculate monthly average
            tsampling = "time.month"   
            #v = "Wstd"
            Wp= W.resample(time="M", skipna =  True).mean() 
            Wp = set_atts_all_vars(Wp, fill_value=fill_value)
            enc = {}
            for v in Wp.data_vars:
                enc[v] = {"dtype": "float32", "_FillValue": fill_value}

         
           
            print('Wp', Wp)
            
            Wp.to_netcdf(tit, mode = "w", encoding=enc)
            gc.collect()
    
   
