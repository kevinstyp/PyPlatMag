import logging
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_model(model_input_train, y_train, model_input_test, y_test, model, param, model_name,
                   year_month_specifiers, learn_config):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])

    # run prediction
    predictions_train = model.predict(model_input_train)
    predictions_test = model.predict(model_input_test)

    # calculate residuals
    mae_train = mean_absolute_error(y_train, predictions_train, batch_size=learn_config.batch_size)
    mae_test = mean_absolute_error(y_test, predictions_test, batch_size=learn_config.batch_size)
    mse_train = mean_squared_error(y_train, predictions_train, batch_size=learn_config.batch_size)
    mse_test = mean_squared_error(y_test, predictions_test, batch_size=learn_config.batch_size)
    std_train = np.std(y_train - predictions_train)
    std_test = np.std(y_test - predictions_test)

    csv_format_string = f"{year_months},{mae_train},{mse_train},{std_train},{mae_test},{mse_test},{std_test},"

    # Generate log
    logger.info(f"Model evaluation: \n{csv_format_string}")

    return csv_format_string