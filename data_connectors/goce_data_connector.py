import logging
import os
import time

import pandas as pd

from data_connectors.data_connector import Connector

logger = logging.getLogger(__name__)

#TODO: Clean up in this class, spend some code comments
class GOCEConnector(Connector):
    def __init__(self, base_path):
        self.data = None

        self.many_params = {
            0: "GO_data_",
            1: "GO_rawdata_",
            2: "GO_rawdata_hk_",
            3: "GO_rawdata_telemetry_"
        }
        self.base_name1 = "GO_data_"
        self.base_name2 = ".dat.gz"
        self.base_name3 = ".h5"
        
        self.base_path = base_path

    # TODO: This method was just copied, no further optimization
    def get_data(self, year_month_specifier):
        start_overall = time.process_time()
        # 3 files to read in
        for i in range(4):
            h5 = False
            if os.path.isfile(self.base_path + self.many_params.get(i) + year_month_specifier + self.base_name3):
                whole_file_name = self.base_path + self.many_params.get(i) + year_month_specifier + self.base_name3
                h5 = True
            else:
                whole_file_name = self.base_path + self.many_params.get(i) + year_month_specifier + self.base_name2

            logger.info(f"reading CSV: {whole_file_name}")

            if i == 1:
                if h5:
                    iter_data = pd.read_hdf(whole_file_name, "df")
                else:
                    iter_data = pd.read_csv(whole_file_name, compression="gzip", sep=',')
            else:
                if h5:
                    iter_data = pd.read_hdf(whole_file_name, "df")
                else:
                    iter_data = pd.read_csv(whole_file_name, index_col=0, compression="gzip", sep=',')
            logger.info(f"data chunk #{str(i)}: {str(iter_data.shape)}")
            logger.debug(f"columns of data chunk: {iter_data.columns.tolist()}")
            if i == 0:
                data = iter_data
            else:
                gps_sec = False
                egg_iaq_index = False
                if i == 1:  # 'gps_sec' in iter_data.columns:
                    gps_sec = True
                elif 'egg_iaq_index' in iter_data.columns:
                    egg_iaq_index = True
                cols_to_use = iter_data.columns.difference(data.columns)
                logger.debug(f"Newly added columns from data chunk: {cols_to_use}")
                if gps_sec:
                    cols_to_use = cols_to_use.append(pd.Index(['gps_sec']))
                if egg_iaq_index:
                    cols_to_use = cols_to_use.append(pd.Index(['egg_iaq_index']))
                logger.debug(f"egg_iaq_index: {egg_iaq_index}")
                logger.debug(f"gps_sec: {gps_sec}")

                if i == 1:
                    data['copy_egg_iaq_index'] = iter_data.index
                    data = data.merge(iter_data[cols_to_use], left_on="gps_sec", right_on="gps_sec",
                                      how="left")
                    data = data.rename(columns={'copy_egg_iaq_index': 'egg_iaq_index'}).set_index('egg_iaq_index')
                else:
                    ##Needed only for last one but should do no harm to others, I hope
                    iter_data.index = pd.to_datetime(iter_data.index)
                    data = data.join(iter_data[cols_to_use], how='left')

            logger.info(f"Current data shape: {data.shape}")
            logger.info(f"\n---\n")

        ### Renaming to my conventions
        # 'qdlat' -> 'APEX_QD_LAT'
        # TODO: Why?? Isn't qdlat the much nicer name??
        data = data.rename(columns={"qdlat": "APEX_QD_LAT"})
        data = data.rename(columns={"qdlon": "APEX_QD_LON"})
        # MLT
        data = data.rename(columns={"mlt": "APEX_MLT"})
        # Longitude
        data = data.rename(columns={"lon.trs": "RAW_Longitude"})
        # Longitude
        data = data.rename(columns={"lat.trs": "RAW_Latitude"})
        # chaos7_b_nec_x -> CHAOS_B_FGM1_1
        data = data.rename(columns={"chaos7_b_fgm1_x": "CHAOS_B_FGM1_0", "chaos7_b_fgm1_y": "CHAOS_B_FGM1_1",
                                    "chaos7_b_fgm1_z": "CHAOS_B_FGM1_2"})

        logger.info(f"Time for input connector: {str(round(time.process_time() - start_overall, 2))} seconds")
        return data
