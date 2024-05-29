import requests
from datetime import datetime

def fetch_omni_data(year, outdir, data_spec="hourly"):
    """
    Fetches OMNI data for a given year.

    Args:
    year (str): The year in 'yyyy' format.

    Returns:
    None
    """
    print(year)

    url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    startday = "0101"

    if datetime.now().year == int(year):
        endday = datetime.now().strftime("%m%d")
        print(endday)
    else:
        endday = "1231"

    startdate = f"{year}{startday}"
    enddate = f"{year}{endday}"

    print(f"{startdate} to {enddate}")


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
        outfile = f"omni2_kp_ae_dst_f107_{year}.lst"
    elif data_spec == "minute":
        # Amps Params
        # vars 15, 16, 21 for  By, Bz, Vsw
        id_By = "15"
        id_Bz = "16"
        id_Vsw = "21"
        params["vars"] = [id_By, id_Bz, id_Vsw]
        params["res"] = "min"
        params["spacecraft"] = "omni_min"
        outfile = f"omni_min_by_bz_vsw_{year}.lst"



    response = requests.get(url, params=params, allow_redirects=True)
    #response = requests.post(url, params=params, allow_redirects=True, headers=headers)
    print("response.request: ", response.request)
    print("response.request.headers: ", response.request.headers)
    print("response.request.body: ", response.request.body)
    print("response.request.url: ", response.request.url)
    print("response.request.method: ", response.request.method)
    print("response: ", response)
    print("response.text: ", response.text)
    print("response.content: ", response.content)
    print("response.headers: ", response.headers)





    if response.status_code == 200:
        with open(outfile, "w") as f:
            f.write(response.text)
    else:
        print("Failed to retrieve data")
        return

    with open(outfile, "r") as f:
        lines = f.readlines()

    # Remove HTML tagged lines, empty lines, parameter lines, and header line
    cleaned_lines = [
        line for line in lines if line.startswith(year)
        #if not (line.strip().startswith("<") or line.strip() == "" or "YEAR" in line)
    ]

    with open(outfile, "w") as f:
        f.writelines(cleaned_lines)


# Example usage:
fetch_omni_data("2021", "omnidata")  # Or fetch_omni_data("2021") for the whole year
