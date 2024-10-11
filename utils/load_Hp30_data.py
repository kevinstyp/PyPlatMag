import logging
import os
from datetime import datetime

import pandas as pd

from lib.get_Kp_index import getKpindex

logger = logging.getLogger(__name__)


def get_output_filename(year, outdir="./data/auxiliary_params/"):
    return f"{outdir}hp30_{year}.h5"


def fetch_Hp30_data(year, outdir="./data/auxiliary_params/"):
    logger.debug(f"year: {year}")
    startday = "01-01"
    if datetime.now().year == int(year):
        endday = datetime.now().strftime("%m-%d")
        logger.debug(f"endday: {endday}")
    else:
        endday = "12-31"

    startdate = f"{year}-{startday}"
    enddate = f"{year}-{endday}"
    logger.debug(f"{startdate} to {enddate}")

    time_Hp30, index_Hp30, status_Hp30 = getKpindex(startdate, enddate, 'Hp30')
    #time_Ap, index_Ap, status_Ap = getKpindex(startdate, enddate, 'Ap')

    hp30_df = pd.DataFrame({'HP_Timestamp': pd.to_datetime(time_Hp30).tz_localize(None), 'Hp30': index_Hp30})
    logger.debug(f"hp30_df.shape: {hp30_df.shape}")
    outfile = get_output_filename(year, outdir)
    logger.debug(f"outfile: {outfile}")

    # mkdir outdir if not exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # write to h5 file
    hp30_df.to_hdf(outfile, key='df', mode='w')

    pass
