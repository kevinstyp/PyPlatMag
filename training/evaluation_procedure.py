import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def evaluate_model(model_input_train, y_train, model_input_test, y_test, model,
                   year_month_specifiers, learn_config, number_of_bisa_neurons=0):
    year_months = '_'.join([year_month_specifiers[0], year_month_specifiers[-1]])

    # run prediction
    predictions_train = model.predict(
        [model_input_train.iloc[:, :-number_of_bisa_neurons]] + [model_input_train.iloc[:, i] for i in
                                                                 range(model_input_train.shape[1] - number_of_bisa_neurons,
                                                                       model_input_train.shape[1])],
        batch_size=learn_config.batch_size)
    predictions_test = model.predict(
        [model_input_test.iloc[:, :-number_of_bisa_neurons]] + [model_input_test.iloc[:, i] for i in
                                                                range(model_input_test.shape[1] - number_of_bisa_neurons,
                                                                        model_input_test.shape[1])],
        batch_size=learn_config.batch_size)

    # calculate residuals
    mae_train = mean_absolute_error(y_train, predictions_train)
    mae_test = mean_absolute_error(y_test, predictions_test)
    mse_train = mean_squared_error(y_train, predictions_train)
    mse_test = mean_squared_error(y_test, predictions_test)
    std_train = np.mean(np.std(y_train - predictions_train))
    std_test = np.mean(np.std(y_test - predictions_test))

    csv_format_string = f"{year_months},{mae_train},{mse_train},{std_train},{mae_test},{mse_test},{std_test},"

    # Generate log
    logger.info(f"Model evaluation: \n{csv_format_string}")

    return csv_format_string