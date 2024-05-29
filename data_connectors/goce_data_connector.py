import logging
import os
import time

import pandas as pd

from data_connectors.data_connector import Connector

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

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

            # if i == 0:
            #     iter_data = pd.read_csv(whole_file_name, index_col=0)
            # else:
            #     iter_data = pd.read_csv(whole_file_name)
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
            print("i: " + str(i) + "iter_data: " + str(iter_data.shape))
            print(iter_data.head(2))
            print(iter_data.columns.tolist())
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
                # dfNew = merge(df, df2[cols_to_use], left_index=True, right_index=True, how='outer')
                print("cols_to_use bef:", cols_to_use)
                # print("cols_to_use bef:", (gps_sec, egg_iaq_index))
                if gps_sec:
                    cols_to_use = cols_to_use.append(pd.Index(['gps_sec']))
                if egg_iaq_index:
                    cols_to_use = cols_to_use.append(pd.Index(['egg_iaq_index']))
                logger.debug(f"egg_iaq_index: {egg_iaq_index}")
                logger.debug(f"gps_sec: {gps_sec}")
                # maybe add egg_iaq_index?
                # print("cols_to_use aft:", cols_to_use)
                # print("both data before: ")
                # print(data.head(2))
                # print(iter_data[cols_to_use].head(2))
                if i == 1:
                    print("data-head before first merge: ", data.head(2))

                    # data['copy_egg_iaq_index'] = data.index
                    data['copy_egg_iaq_index'] = iter_data.index
                    # data = data.merge(iter_data[cols_to_use], left_on="gps_sec", right_on="gps_sec", how="right")#on="egg_iaq_index")
                    data = data.merge(iter_data[cols_to_use], left_on="gps_sec", right_on="gps_sec",
                                      how="left")  # on="egg_iaq_index")
                    # print("data-head before index reset", data.head(2))
                    # print("data-columns before index reset", data.columns)
                    data = data.rename(columns={'copy_egg_iaq_index': 'egg_iaq_index'}).set_index('egg_iaq_index')
                    # print("data-head after iaq", data.head(2))
                    # print("data-head before columns", data.columns.tolist())
                    # data
                    # data = data.set_index("egg_iaq_index")
                    # data = data.set_index("gps_sec")
                else:
                    # print("data.index: ", data.index)
                    # print("iter_data[cols_to_use].index: ", iter_data[cols_to_use].index)
                    # data = data.merge(iter_data[cols_to_use], on="egg_iaq_index")

                    ##Needed only for last one but should do no harm to others, I hope
                    iter_data.index = pd.to_datetime(iter_data.index)

                    # print("data-head before join", data.head(2))
                    # print("cols_to_use: ", cols_to_use)
                    # print("iter_data[cols_to_use].head(2): ", iter_data[cols_to_use].head(2))
                    # print("Who are the indices?: ", data.index)
                    # print("Who are the indices?: ", iter_data[cols_to_use].index)
                    # data = data.join(iter_data[cols_to_use], how='left')
                    data = data.join(iter_data[cols_to_use], how='left')
                    # print("data-head after join", data.head(2))
                # print("data-head", data.head(2))

            print("data-shape: ", data.shape)
            print("\n\n---\n\n")
        print("data: ", data.shape)
        print("data: ", data.head(3))

        ### Renaming to my conventions
        # 'qdlat' -> 'APEX_QD_LAT'
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

        logger.info("Time for input connector: " + str(round(time.process_time() - start_overall, 2)) + " seconds")
        print("input co")
        return data
