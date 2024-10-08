import numpy as np


def add_weights(df):
    weightings = np.ones(df.shape[0])
    weightings[np.logical_and(-60 < df["APEX_QD_LAT"], df["APEX_QD_LAT"] < 60)] = 1.5
    weightings[np.logical_and(-50 < df["APEX_QD_LAT"], df["APEX_QD_LAT"] < 50)] = 2.
    df = df.assign(weightings=weightings)
    return df
