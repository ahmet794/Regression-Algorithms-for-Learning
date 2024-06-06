# data_utils.py the functions for processing data, and calling data.

import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def sum_x_y(data):
    '''
    Sums the product of x and y column's values.

    :param data: two dimensional array with two columns.

    :return: Sum of all the product values.
    '''
    total = sum(data[:, 0]*data[:, 1])
    return total



def sum_x(data):
    '''
    Sum the x column values.

    :param data: two dimensional array with two columns.

    :return: Sum of all the x values.
    '''
    total = sum(data[:, 0])
    return total


def sum_y(data):
    """
    Sum the y column values.

    :param data: two dimensional array with two columns.

    :return: Sum of all the y values.
    """
    total = sum(data[:, 1])
    return total


def sumsq_x(data):
    """
    Sum the squaredx column values.

    :param data: two dimensional array with two columns.

    :return: Sum of all the squared x values.
    """
    total = sum((data[:, 0]**2))
    return total


def load_extended_boston(r_state):
    """
    Load the boston house extended dataset from mglearn with a specific random state.

    :param r_state: An integer variable for random state.

    :return: A training data and a test data for developing models.
    """
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data

def load_boston(r_state):
    """
    Load the boston house dataset with a specific random state.

    :param r_state: An integer variable for random state.

    :return: A training and a test data for developing models.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data


def load_california(r_state):
    """
    Load the california housing dataset from sklearn with a specific random state.

    :param r_state: An integer variable for random state.

    :return: A training data and a test data for developing models.
    """
    california_housing = fetch_california_housing()
    X = california_housing.data
    y = california_housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data

def load_washington(r_state):
    """
    Load the washington housing dataset from sklearn with a specific random state.

    :param r_state: An integer variable for random state.

    :return: A training data and a test data for developing models.
    """
    df = pd.read_csv('housingprices_data_kaggle.csv.xls')
    df = df.drop(columns=["date", "country", "statezip", "city", "street"], axis=1)
    data = df.to_numpy()
    data = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]]
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data

def load_newyork(r_state):
    """
    Load the New York housing dataset from sklearn with a specific random state.

    :param r_state: An integer variable for random state.

    :return: A training data and a test data for developing models.
    """
    df = pd.read_csv('NewYorkHousing.csv')
    data = df.to_numpy()
    data = data[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]]
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data

def normalize(train_data, test_data):
    scaler = StandardScaler()
    X_train, Y_train = scaler.fit_transform(train_data[:, :-1]), train_data[:, -1] 
    X_cal, Y_cal = scaler.transform(test_data[:, :-1]), test_data[:, -1]

    train_data = np.hstack((X_train, Y_train.reshape(-1, 1)))
    test_data = np.hstack((X_cal, Y_cal.reshape(-1, 1)))

    return train_data, test_data


# StandardScalar class has methods for normalizing the data.
class StandardScaler():
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, data):
        """
        Calculate the mean and the standard deviation given data.

        :param data: numpy array.
        """
        self.means = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

        self.std[self.std == 0] = 1

    def transform(self, data):
        """
        Normalize the data using the mean and the standard deviation estimated using the fit method.

        :param data: numpy array to be transformed.

        :return: The normalized data as a numpy array.
        """
        return (data - self.means)/self.std

    def fit_transform(self, data):
        """
        Fit to the data then transform it.

        :param data: numpy array to be fit and transformed

        :return: The normalized data as a numpy array.
        """
        self.fit(data)
        return self.transform(data)
 
