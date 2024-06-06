import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

'''
Sum a and b.

:param a: an int or float.
:param b: an int or float.

:return: Sum of a and b.

'''

def add (a, b):
    return a + b


'''
Sums the product of x and y column's values.

:param data: two dimensional array with two columns.

:return: Sum of all the product values.

'''

def sum_x_y(data):
    total = sum(data[:, 0]*data[:, 1])
    return total


'''
Sum the x column values.

:param data: two dimensional array with two columns.

:return: Sum of all the x values.

'''
def sum_x(data):
    total = sum(data[:, 0])
    return total

"""
Sum the y column values.

:param data: two dimensional array with two columns.

:return: Sum of all the y values.
"""
def sum_y(data):
    total = sum(data[:, 1])
    return total

"""
Sum the squaredx column values.

:param data: two dimensional array with two columns.

:return: Sum of all the squared x values.
"""
def sumsq_x(data):
    total = sum((data[:, 0]**2))
    return total

"""
Find the value of intercept for our best fit line.

:param data: two dimensional array with two columns

:return: The value of the intercept.
"""
def intercept(data):
    n = data.shape[0]
    intercept = ((sum_y(data))-slope(data)*sum_x(data))/n
    return round(intercept, 2)

"""
Find the value of the slope for our best fit line.

:param data: two dimensional array with two columns

:return: The value of the slope.
"""
def slope(data):
    n = data.shape[0]
    slope = (n*(sum_x_y(data))-(sum_x(data))*(sum_y(data)))/((n*(sumsq_x(data)))-(sum_x(data))**2)
    return round(slope, 2)

"""
Find the best fit line equation.

:param a: the slope
:param b: the intercept

:return: The linear equation of the best fit line.
"""
def line(a, b):
    return "y="+str(b)+"x"+str(a)

"""
Plot the linear regression using a simple data.

:param data: two dimensional array with two columns

:return: The plot of the graph with data points and the best fit line.
"""
def plot_linear_r(data):
    x = data[:, 0]
    y = data[:, 1]

    b = slope(data)
    a = intercept(data)

    x_line = np.linspace(min(x), max(x), 100) # Create a space so that the line can be formed.
    y_line = b * x_line + a

    plt.scatter(x, y, color='blue', label='Data Points')# Plot data points.
    plt.plot(x_line, y_line, color='red', label=f'y = {b}x + {a}')# Plot the line.
    plt.xlabel("Temperature")
    plt.ylabel("Percentage of Fire")
    plt.legend()
    plt.title('Simple Linear Regression Plot')
    plt.grid(True)
    plt.show()


"""
Using Linear Regression train and make predictions from data.

:param train: A dataset to train and create the best fit line for linear regression.
:param test A dataset to predict the y values.

:return: Return an array of predictions.
"""
def slr(train, test):
    predictions = np.array([])
    intercept_ = intercept(train)
    slope_ = slope(train)
    
    # Train and asses the model for training dataset.
    yhat = intercept_ + slope_ * train[:, 0]
    train_predictions = np.append(predictions, yhat)
    rss_train, tss_train = rss_tss(train, train_predictions)

    # Test the model and asses using the test dataset.
    yhat = intercept_ + slope_ * test[:, 0]
    test_predictions = np.append(predictions, yhat)
    rss_test, tss_test = rss_tss(test, test_predictions)

    r_sq_train = r_squared(rss_train, tss_train)
    r_sq_test = r_squared(rss_test, tss_test)

    results = {
        "Intercept": intercept_,
        "Slope": slope_,
        "Predictions": test_predictions,
        "Training R-Squared": r_sq_train,
        "Test R-Squared": r_sq_test
    }

    return results


"""
Get the r-squared value for a linear model.

:param rss: The residual sum of squares.
:param tss: The total sum of squares.

return: The R-Squared value using the formula.
"""
def r_squared(rss, tss):
    return 1 - (rss/tss)

"""
Get the RSS and TSS of a data and a prediction set.

:param data: Two dimensional dataset.
:param predictions: List of predictions that were made using the model.

:return: The RSS and the TSS.
"""
def rss_tss(data, predictions):
    rss = np.sum((data[:, -1] - predictions)**2)
    tss = np.sum((data[:, -1] - np.mean(data[:, -1]))**2)
    return rss, tss

'''
Train a multiple linear regression model.

:param train: Training dataset.

:return: The beta hat and the r-squared for training set.
'''
def train_mlr(train):
    predictions = np.array([])
    Y = train[:, -1]
    Z = train[:, :-1]
    Z = np.column_stack([np.ones(Z.shape[0]), Z])

    beta_hat = np.linalg.pinv(Z.T @ (Z)) @ (Z.T) @ (Y)
    Y_hat = Z @ (beta_hat) 
    predictions = np.append(predictions, Y_hat)

    rss, tss = rss_tss(train, predictions)
    r_sq = r_squared(rss, tss)
    variance = var(beta_hat)

    results = {
        "Beta_hat": beta_hat,
        "R-Squared Training": r_sq,
        "Variance": variance
    }
    return results

"""
Test a multiple linear regression model.

:param test: A test dataset.
:param beta_hat: Beta hat value from the training model.

:return: The prediction set and the r-squared for the test set.
"""
def predict_mlr(test, beta_hat):
    predictions = np.array([])
    Z = test[:, :-1]
    Z = np.column_stack([np.ones(Z.shape[0]), Z])

    Y_hat = Z.dot(beta_hat) 
    predictions = np.append(predictions, Y_hat)

    rss, tss = rss_tss(test, predictions)
    r_sq = r_squared(rss, tss)
    variance = var(beta_hat)

    results = {
        "Predictions": predictions,
        "R-Squared Test": r_sq,
        "Variance": variance
    }
    return results


"""
Train and create a ridge regression model.

:param train: A training dataset where the response variable is the last column.
:param alpha: The tuning parameter.

:return: A dictionary consisting of coefficients, R-Squared Training and the Variance.
"""
def train_ridge_regression(train, alpha=1.0):
    Y = train[:, -1]
    Z = train[:, :-1]
    Z = np.column_stack([np.ones(Z.shape[0]), Z])

    I_p = np.eye(Z.shape[1])
    I_p[0, 0] = 0
    
    beta_hat = np.linalg.pinv(Z.T.dot(Z) + alpha * I_p).dot(Z.T).dot(Y)
    predictions = Z.dot(beta_hat)

    rss, tss = rss_tss(train, predictions)
    r_sq = r_squared(rss, tss)
    variance = var(beta_hat)

    results = {
        "Beta Hat": beta_hat,
        "R-Squared Training": r_sq,
        "Variance": variance
    }

    return results


"""
Testing an existing ridge model.

:param test: A test set that has the last columns as the response variable.
:param beta_hat: A list of coefficients from the trained ridge model.

:return: A dictionary with, Predictions, R-Squared, and Variance of the tested data.

"""
def test_ridge_regression(test, beta_hat):
    predictions = np.array([])
    Z = test[:, :-1]
    Z = np.column_stack([np.ones(Z.shape[0]), Z])
    
    Y_hat = Z.dot(beta_hat)
    predictions = np.append(predictions, Y_hat)

    rss, tss = rss_tss(test, predictions)
    r_sq = r_squared(rss, tss)
    variance = var(beta_hat)

    results = {
        "Predictions": predictions,
        "R-Squared Test": r_sq,
        "Variance": variance
    }

    return results

"""
Sample variance of the coefficients.

:param coefficients: A list of coefficients.

:return: The variance between the coefficients.
"""
def var(coefficients):
    return np.var(coefficients, ddof=1)

"""
Load the boston house extended dataset from mglearn with a specific random state.

:param r_state: An integer variable for random state.

:return: A training data and a test data for developing models.
"""
def load_boston(r_state):
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data

"""
Load the california housing dataset from sklearn with a specific random state.

:param r_state: An integer variable for random state.

:return: A training data and a test data for developing models.
"""
def load_california(r_state):
    california_housing = fetch_california_housing()
    X = california_housing.data
    y = california_housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=r_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    train_data = np.concatenate((X_train, y_train), axis=1)
    test_data = np.concatenate((X_test, y_test), axis=1)
    return train_data, test_data


"""
Plot a comparison graph of LR and RR coefficients.

:param lr_coefs: A list of LR model coefficients.
:param r_coefs: A list of RR model coefficients.
:param alpha: A float for the tuning parameter.

:return: Shows the bar graph.
"""
def plt_ridge_lr_comparison(lr_coefs, r_coefs, alpha):
    indices = np.arange(len(lr_coefs))

    plt.figure(figsize=(12, 6))
    width = 0.35

    plt.bar(indices - width/2, lr_coefs, width=width, label="Linear Regression Ceofficients", color = 'blue')
    plt.bar(indices + width/2, r_coefs, width=width, label='Ridge Regression Coefficients', color = "green")

    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title(f"Comparison of Linear Regression and Ridge Regression Coefficients, $\lambda={alpha}$")
    plt.legend()

    plt.show()














