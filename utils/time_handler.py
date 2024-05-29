from astropy.time import Time


def gps_to_datetime(gps_secs):
    t = Time(gps_secs, format='gps', scale='utc')
    return t.datetime


def datetime_to_gps(datetimes):
    # add Gps-Seconds to the Gps-Epoch and convert with units.second
    t = Time(datetimes, scale='utc')
    return t.gps
