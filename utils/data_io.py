import os


def save_df(data, write_path, satellite_specifier, year_month_specifier):
    path = write_path + satellite_specifier + "/" + year_month_specifier
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_hdf(path + "/data.h5", key='df', mode='w')
