import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

def get_output_filename(year, data_spec="hourly"):
    if data_spec == "hourly":
        outfile = f"omni2_kp_dst_f107_{year}.lst"
    elif data_spec == "minute":
        outfile = f"omni_min_by_bz_vsw_{year}.lst"
    else:
        raise ValueError("data_spec must be 'hourly' or 'minute'")
    return outfile

# TODO: Why this does not use outdir here? Where is the file written to?
def fetch_omni_data(year, outdir, data_spec="hourly"):
    """
    Fetches OMNI data for a given year.

    Args:
    year (str): The year in 'yyyy' format.

    Returns:
    None
    """
    logger.debug(f"year: {year}")

    url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    startday = "0101"

    if datetime.now().year == int(year):
        endday = datetime.now().strftime("%m%d")
        logger.debug(f"endday: {endday}")
    else:
        endday = "1231"

    startdate = f"{year}{startday}"
    enddate = f"{year}{endday}"
    logger.debug(f"{startdate} to {enddate}")

    params = {
        'activity': 'retrieve',
        'start_date': startdate,
        'end_date': enddate,
    }

    if data_spec == "hourly":
        # Kp / Dst Params
        # vars 38, 40, 50 for Kp, Dst, F10.7
        id_Kp = "38"
        id_Dst = "40"
        id_F107 = "50"
        params["vars"] = [id_Kp, id_Dst, id_F107]
        params["res"] = "hour"
        params["spacecraft"] = "omni2"
    elif data_spec == "minute":
        # Amps Params
        # vars 15, 16, 21 for  By, Bz, Vsw
        id_By = "17"
        id_Bz = "18"
        id_Vsw = "21"
        params["vars"] = [id_By, id_Bz, id_Vsw]
        params["res"] = "min"
        params["spacecraft"] = "omni_min"

    outfile = get_output_filename(year, data_spec)

    response = requests.get(url, params=params, allow_redirects=True)
    if response.status_code == 200:
        with open(outfile, "w") as f:
            f.write(response.text)
    else:
        # Throw exception if request fails
        logger.error(f"Failed to retrieve omni data for year {year} with status code {response.status_code}")
        raise Exception(f"Failed to retrieve omni data for year {year} with status code {response.status_code}")

    with open(outfile, "r") as f:
        lines = f.readlines()

    # Remove HTML tagged lines, empty lines, parameter lines, and header line
    cleaned_lines = [
        line for line in lines if line.startswith(year)
    ]

    with open(outfile, "w") as f:
        f.writelines(cleaned_lines)
