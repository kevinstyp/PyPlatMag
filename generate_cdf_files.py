import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml
from box import Box

import publication.training_retrieval as tr
from data_filters.goce_filter import goce_filter
from training import training_data, training_procedure as tp
from utils import data_io, quaternion_util as qu
from utils import cdf_util as cu

dirname = os.path.dirname(__file__)
config = Box.from_yaml(filename=os.path.join(dirname, "./config.yaml"), Loader=yaml.SafeLoader)
config_goce = Box.from_yaml(filename=os.path.join(dirname, "./config_goce.yaml"), Loader=yaml.SafeLoader)

os.environ["CDF_LIB"] = config.CDF_LIB
from spacepy import pycdf

logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(config.log_level),
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"config: {config}")
logger.info(f"config_goce: {config_goce}")


# Get the year_months used to save auxilary files
if config.model_year_months == "None":
    year_months = '_'.join([config.year_month_specifiers[0], config.year_month_specifiers[-1]])
else:
    year_months = config.model_year_months

# Load the auxilary files created during training
train_config = config_goce.train_config
std_indices, corr_indices, hk_scaler = tr.read_in_pickles_small(train_config.training_file_path, year_months)

# Specify the model to be loaded
model_name = config.model_output_path + config.model_name + '_' + config.satellite_specifier + '_' + year_months + '.h5'
logger.info(f"model_name: {model_name}")

# Now similar to training
# Load the data
data = data_io.read_df(config.write_path, config.satellite_specifier, config.year_month_specifiers, dataset_name="data_nonan")
logger.info(f"Data shape after reading: {data.shape}")
data = goce_filter(data, magnetic_activity=True, doy=True, training=False, training_columns=[],
            meta_features=config_goce.meta_features, y_features=config_goce.y_all_feature_keys)
logger.info(f"Data shape after filtering: {data.shape}")


# TODO: Check whether something happens because of this
# Leave align_x_columns away?: Columns are sorted -> not necessary
# Weight handling: Weights arent needed anymore -> not necessary
# Check if someone randomizes the columns at some point :O -> not necessary

# Extract power currents if use_pinn is set
train_config = config_goce.train_config
if train_config.use_pinn:
    current_parameters_file = os.path.join(dirname, config.current_parameters_file)
    goce_column_description_file = os.path.join(dirname, config_goce.goce_column_description_file)
    data, electric_current_df = tp.extract_electric_currents(data, config_goce.current_parameters_file,
                                                             config_goce.goce_column_description_file)


# TODO: training_data, training_prcedure -> Maybe, rename them to preprare_data, prepare_procedure or smth
x_all, y_all, z_all, weightings = training_data.split_dataframe(data, config_goce.y_all_feature_keys, config_goce.meta_features)

# Add solar activity, and DOY
x_all = tp.add_solar_activity(x_all, z_all)
x_all = tp.add_day_of_year(x_all, z_all)

training_file_path = os.path.join(dirname, train_config.training_file_path)
# Std, Corr, Scaling
if train_config.filter_std:
    x_all = tp.filter_std(x_all, training_file_path, config.year_month_specifiers, use_cache=True)
    logger.debug(f"x_all - shape after std filtering: {x_all.shape}")

if train_config.filter_correlation:
    x_all = tp.filter_correlation(x_all, training_file_path, config.year_month_specifiers, use_cache=True)
    logger.debug(f"x_all - shape after correlation filtering: {x_all.shape}")

x_all = tp.scale_data(x_all, training_file_path, config.year_month_specifiers, config.use_cache)

logger.info(f"x_all - shape before splitting: {x_all.shape}")
logger.info(f"Final columns for generating predictions: {x_all.columns.tolist()}")

# Check how to split network building and network training etc.
if train_config.use_pinn:
    model_input = pd.concat([x_all, electric_current_df], axis=1)
else:
    model_input = x_all

# TODO: Test that the loaded model really produces the same results, otherwise need to change to "manual" weight loading
from tensorflow import keras
from training.customs.pinn_biot_savart_layer import BiotSavartLayer
from training.customs.custom_initializer import CustomInitializer
model = keras.models.load_model(model_name,
                                 custom_objects={
                                     'CustomInitializer': CustomInitializer,
                                                 'BiotSavartLayer': BiotSavartLayer})
# number_of_bisa_neurons = electric_current_df.shape[1]
# from training.model_builder import build_network_goce_pinn
# import tensorflow as tf
# model = build_network_goce_pinn(input_shape=(model_input.shape[1] - number_of_bisa_neurons),
#                                         #input_shape=(x_train[0].shape[1]),
#                                         batch_size=train_config.learn_config.batch_size,
#                                                 number_of_bisa_neurons=number_of_bisa_neurons,
#                                                  )
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)#, clipnorm=0.1)
# model.compile(loss='mean_squared_error',  # loss='mean_squared_error',
#               optimizer=optimizer,
#               metrics=['mse', 'mae'])
# model.load_weights(model_name)
for layer in model.layers:
    print("layer.name: ", layer.name)
    # print("layer.get_weights(): ", layer.get_weights())
    if "biot_savart_layer_" in layer.name:
        # TODO: Tweak this, so the Radii are ouput
        print("layer.get_weights(): ", layer.get_weights())
        print("Length: ", np.linalg.norm(layer.get_weights()[0]))

learn_config = train_config.learn_config
number_of_bisa_neurons = electric_current_df.shape[1]
print(f"model_input.shape: {model_input.shape}")
print(f"model_input.columns: {model_input.columns.tolist()}")
predictions_fgm = model.predict(
    [model_input.iloc[:, :-number_of_bisa_neurons]] + [model_input.iloc[:, i] for i in range(model_input.shape[1] - number_of_bisa_neurons, model_input.shape[1])],
                                batch_size=learn_config.batch_size)

del model_input, x_all, model

# TODO: To which extend can this be put into a function?
quaternions = z_all[['q1_fgm12nec', 'q2_fgm12nec', 'q3_fgm12nec', 'q4_fgm12nec']].values
predictions_nec = qu.rotate_mag2nec(quaternions, predictions_fgm)

## Flag generation
b_flag = np.zeros_like(predictions_fgm[:,0])
b_flag = b_flag + z_all['Outlier_Flag'].values * 1
b_flag = b_flag + z_all['Interpolation_Distance_Flag'].values * 2
b_flag = b_flag + z_all['Spaceweather_Flag'].values * 4

magnetic_activity_flag = z_all['Magnetic_Activity_Flag'].values
nan_flag = z_all['NaN_Flag'].values

# Start with CDF-creation
cdf_config = config_goce.cdf_config

out_params = ["calcorr", "chaos"]


base_out_path = cdf_config.cdf_path + config.satellite_specifier + '/cal_pinn/'

timestamp_series = pd.DatetimeIndex(z_all["RAW_Timestamp"])
print("timestamp_series: ", timestamp_series)

all_dates = np.array(list(
    map(lambda y, m, d: '{:04d}{:02d}{:02d}'.format(y, m, d), timestamp_series.year, timestamp_series.month,
        timestamp_series.day))).astype(int)
unique_dates = np.sort(np.unique(all_dates))
print("unique_dates: ", unique_dates)

y_all = y_all.values

from sklearn.metrics import mean_absolute_error, mean_squared_error
meanae = mean_absolute_error(y_all[~magnetic_activity_flag], predictions_fgm[~magnetic_activity_flag])
meanse = mean_squared_error(y_all[~magnetic_activity_flag], predictions_fgm[~magnetic_activity_flag])
std1 = np.std(y_all[~magnetic_activity_flag] - predictions_fgm[~magnetic_activity_flag])
print('mae NN - train: ', meanae)
print('mse NN - train: ', meanse)
print("std NN: ", std1)

cdf_template_path = os.path.join(dirname, cdf_config.cdf_template_path)
for act_date in sorted(unique_dates):
    sel = (all_dates == act_date)
    logger.info(f"{np.sum(sel)} samples for date {act_date}")
    if np.sum(sel) > 1:
        # Otherwise problems with pandas returning float object instead of list/array for sel
        # Can happen e.g. if one timestamp is found in wrong month somehow

        for conf_out_param in out_params:
            # Acal_Corr
            if conf_out_param == 'calcorr':
                output = {}
                output['Timestamp'] = timestamp_series[sel]
                output['Latitude'] = z_all["RAW_Latitude"].values[sel]
                output['Longitude'] = z_all["RAW_Longitude"].values[sel]
                output['QDLatitude'] = z_all["APEX_QD_LAT"].values[sel]
                output['QDLongitude'] = z_all["APEX_QD_LON"].values[sel]
                # Conversion to meters in saved CDF files
                output['Radius'] = z_all["r.trs"].values[sel] * 1000.
                output['B_MAG'] = np.c_[predictions_fgm[:, 0][sel], predictions_fgm[:, 1][sel], predictions_fgm[:, 2][sel]]
                output['B_NEC'] = np.c_[predictions_nec[:, 0][sel], predictions_nec[:, 1][sel], predictions_nec[:, 2][sel]]

                output['q_MAG_NEC'] = np.c_[
                    quaternions[:, 0][sel], quaternions[:, 1][sel], quaternions[:, 2][sel], quaternions[:, 3][sel]]
                output['B_FLAG'] = b_flag[sel]
                output['NaN_FLAG'] = nan_flag[sel]
                output['MAGNETIC_ACTIVITY_FLAG'] = magnetic_activity_flag[sel]
                name_part = '_MAG_ACAL_CORR_ML_'
                master_cdf_path = cdf_template_path + cdf_config.version + '/template_aligncal_corr_v' + cdf_config.version + '.cdf'

            if conf_out_param == 'chaos':
                output = {}
                output['Timestamp'] = timestamp_series[sel]
                print("output['Timestamp']: ", output['Timestamp'])

                output['B_MAG'] = np.c_[y_all[:, 0][sel], y_all[:, 1][sel], y_all[:, 2][sel]]

                print("output['B_MAG']: ", output['B_MAG'][:10])
                output['B_NEC'] = np.c_[z_all['chaos7_b_nec_x'].values[sel], z_all['chaos7_b_nec_y'].values[sel],
                                        z_all['chaos7_b_nec_z'].values[sel]]

                # Also add Amps model values for MAG and NEC
                output['B_AMPS_MAG'] = np.c_[z_all['amps_b_mag_x'].values[sel], z_all['amps_b_mag_y'].values[sel],
                                             z_all['amps_b_mag_z'].values[sel]]
                output['B_AMPS_NEC'] = np.c_[z_all['amps_b_nec_x'].values[sel], z_all['amps_b_nec_y'].values[sel],
                                             z_all['amps_b_nec_z'].values[sel]]

                output['B_COR_NEC'] = np.c_[
                    z_all['chaos7_b_cor_nec_x'].values[sel], z_all['chaos7_b_cor_nec_y'].values[sel],
                    z_all['chaos7_b_cor_nec_z'].values[sel]]
                output['B_LIT_NEC'] = np.c_[
                    z_all['chaos7_b_lit_nec_x'].values[sel], z_all['chaos7_b_lit_nec_y'].values[sel],
                    z_all['chaos7_b_lit_nec_z'].values[sel]]
                output['B_EXT_NEC'] = np.c_[
                    z_all['chaos7_b_ext_nec_x'].values[sel], z_all['chaos7_b_ext_nec_y'].values[sel],
                    z_all['chaos7_b_ext_nec_z'].values[sel]]
                name_part = '_MODEL_CHAOS7_'
                master_cdf_path = cdf_template_path + cdf_config.version + '/template_chaos7_v' + cdf_config.version + '.cdf'

            # Now the writing itself
            str_start = timestamp_series[sel].min().floor('1s').strftime('%Y%m%dT%H%M%S')
            str_stop = timestamp_series[sel].max().floor('1s').strftime('%Y%m%dT%H%M%S')
            out_path = base_out_path + conf_out_param + '/v' + cdf_config.version + '/'
            cdffilename = out_path + cdf_config.sat_suffix + name_part + str_start + '_' + str_stop + '_' + \
                          cdf_config.version + '.cdf'
            if os.path.isfile(cdffilename):
                os.remove(cdffilename)
            if not os.path.isfile(cdffilename):
                if not os.path.isdir(os.path.dirname(cdffilename)):
                    os.makedirs(os.path.dirname(cdffilename))
                print(cdffilename)
                print("master_cdf_path: ", master_cdf_path)
                if not os.path.isfile(master_cdf_path):
                    logger.warning(f"Master CDF file {master_cdf_path} not found. Creating from template.")
                    cu.create_mastercdf(master_cdf_path)

                cdffile = pycdf.CDF(cdffilename, masterpath=master_cdf_path)
                cdffile.attrs['TITLE'] = os.path.basename(cdffilename).replace('.cdf', '')
                cdffile.attrs['ORIGINAL_PRODUCT_NAME'] = os.path.basename(cdffilename).replace('.cdf', '')
                cdffile.attrs['Generation_date'] = datetime.datetime.utcnow().strftime("%04Y-%02m-%02d %02H:%02M:%02S")
                cdffile.attrs['DOI'] = ''
                cdffile.attrs['License'] = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
                for key in output.keys():
                    print(key)
                    cdffile[key] = np.squeeze(output[key])
                    try:
                        cdffile.compress(pycdf.const.GZIP_COMPRESSION, 1)
                    except:
                        # workaround if compression fails
                        print('compression failed')
                        cdffile.close()
                        os.remove(cdffilename)
                        cdffile = pycdf.CDF(cdffilename, masterpath=master_cdf_path)
                        cdffile.attrs['TITLE'] = (os.path.split(cdffilename)[-1]).replace('.cdf', '')
                        cdffile.attrs['ORIGINAL_PRODUCT_NAME'] = (os.path.split(cdffilename)[-1]).replace('.cdf', '')
                        cdffile.attrs['Generation_date'] = datetime.datetime.utcnow().strftime("%04Y-%02m-%02d %02H:%02M:%02S")
                        cdffile.attrs['DOI'] = ''
                        cdffile.attrs['License'] = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
                        for key in output.keys():
                            cdffile[key] = np.squeeze(output[key])
                            print(cdffilename + ': no compression applied')
                    if key == 'Timestamp':
                        print("cdffile: ", cdffile['Timestamp'][0])
                cdffile.close()