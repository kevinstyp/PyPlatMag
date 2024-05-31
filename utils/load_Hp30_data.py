from datetime import datetime
from lib.get_Kp_index import getKpindex
import pandas as pd
import os

def get_output_filename(year, outdir="./data/auxiliary_params/"):
    #return f"hp30_{year}.h5"
    return f"{outdir}/hp30_{year}.h5"

def fetch_Hp30_data(year, outdir="./data/auxiliary_params/"):
    print(year)

    startday = "01-01"
    if datetime.now().year == int(year):
        endday = datetime.now().strftime("%m-%d")
        print(endday)
    else:
        endday = "12-31"

    startdate = f"{year}-{startday}"
    enddate = f"{year}-{endday}"

    print(f"{startdate} to {enddate}")


    time_Hp30, index_Hp30, status_Hp30 = getKpindex(startdate, enddate, 'Hp30')
    #time_Ap, index_Ap, status_Ap = getKpindex(startdate, enddate, 'Ap')
    # print("time_Hp30: ", time_Hp30)
    # print("index_Hp30: ", index_Hp30) #Here are the Hp30 values
    # #print("status_Hp30: ", status_Hp30)
    # print("time_Hp30.shape: ", time_Hp30.shape)
    # print("index_Hp30.shape: ", index_Hp30.shape)
    #print("status_Hp30.shape: ", status_Hp30.shape)

    hp30_df = pd.DataFrame({'HP_Timestamp': pd.to_datetime(time_Hp30).tz_localize(None), 'Hp30': index_Hp30})
    print("hp30_df: ", hp30_df)
    print("hp30_df.shape: ", hp30_df.shape)
    print("hp30_df.dtypes: ", hp30_df.dtypes)
    outfile = get_output_filename(year, outdir)

    # mkdir outdir if not exists
    if not os.path.exists(outfile):
        os.makedirs(outfile)

    # write to h5 file
    hp30_df.to_hdf(f"{outfile}", key='df', mode='w')

    pass