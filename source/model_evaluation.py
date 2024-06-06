#model_evaluation.py is dedicated to evaluate the models.

import numpy as np


def r_squared(rss, tss):
    """
    Get the r-squared value for a linear model.

    :param rss: The residual sum of squares.
    :param tss: The total sum of squares.

    return: The R-Squared value using the formula.
    """
    return 1 - (rss/tss)


def rss_tss(data, predictions):
    """
    Get the RSS and TSS of a data and a prediction set.

    :param data: Two dimensional dataset.
    :param predictions: List of predictions that were made using the model.

    :return: The RSS and the TSS.
    """
    rss = np.sum((data[:, -1] - predictions)**2)
    tss = np.sum((data[:, -1] - np.mean(data[:, -1]))**2)
    return rss, tss


def var(coefficients):
    """
    Sample variance of the coefficients.

    :param coefficients: A list of coefficients.

    :return: The variance between the coefficients.
    """
    return np.var(coefficients, ddof=1)

def mse(y, y_hat):
    """
    Mean Squared Error of a model depending on the observed and predicted variables.

    :param y: A list of observed variables.
    :param y_Hat: A list of predicted variables.

    :return: The Mean Squared Error of a model.
    """
    return np.mean((y - y_hat)**2)

