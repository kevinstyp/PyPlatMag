import pickle
import logging

logger = logging.getLogger(__name__)


# TODO: Rename function
def read_in_pickles_small(training_file_path, year_months):
    print("training_file_path: ", training_file_path)
    print("year_months: ", year_months)
    std_file = training_file_path + year_months + "/std_column_indices.pkl"
    corr_file = training_file_path + year_months + "/corr_column_indices.pkl"
    scaler_file = training_file_path + year_months + "/scaler.pkl"

    logger.info(f"scaler_file: {scaler_file}")
    with open(std_file, 'rb') as f:
        std_indices = pickle.load(f)

    with open(corr_file, 'rb') as f:
        corr_indices = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        hk_scaler = pickle.load(f)
    return (std_indices, corr_indices, hk_scaler)

