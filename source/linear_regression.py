#linear_regression.py the simple linear regression and multiple linear regression classes and functions.

import data_utils as d_u
import model_evaluation as m_e
import numpy as np

class SimpleLinearRegression:
    def __init__(self, data):
        self.data = data


    def slope(self, data):
        """
        Find the value of the slope for our best fit line.

        :param data: two dimensional array with two columns

        :return: The value of the slope.
        """
        n = self.data.shape[0]
        slope = (n*(d_u.sum_x_y(data))-(d_u.sum_x(data))*(d_u.sum_y(data)))/((n*(d_u.sumsq_x(data)))-(d_u.sum_x(data))**2)
        return round(slope, 2)



    def intercept(self, data):
        """
        Find the value of intercept for our best fit line.

        :param data: two dimensional array with two columns

        :return: The value of the intercept.
        """
        n = data.shape[0]
        intercept = ((d_u.sum_y(data))-self.slope(data)*d_u.sum_x(data))/n
        return round(intercept, 2)
    

 
    def line(self, a, b):
        """
        Find the best fit line equation.

        :param a: the slope
        :param b: the intercept

        :return: The linear equation of the best fit line.
        """
        return "y="+str(b)+"x"+str(a)
    


    def train_test(self, train, test):
        """
        Using Linear Regression train and make predictions from data.

        :param train: A dataset to train and create the best fit line for linear regression.
        :param test A dataset to predict the y values.

        :return: Return an array of predictions, the intercept, slope, training R-Squared and the Test R-Squared.
        """
        predictions = np.array([])
        intercept_ = self.intercept(train)
        slope_ = self.slope(train)
        
        # Train and asses the model for training dataset.
        yhat = intercept_ + slope_ * train[:, 0]
        train_predictions = np.append(predictions, yhat)
        rss_train, tss_train = m_e.rss_tss(train, train_predictions)

        # Test the model and asses using the test dataset.
        yhat = intercept_ + slope_ * test[:, 0]
        test_predictions = np.append(predictions, yhat)
        rss_test, tss_test = m_e.rss_tss(test, test_predictions)

        r_sq_train = m_e.r_squared(rss_train, tss_train)
        r_sq_test = m_e.r_squared(rss_test, tss_test)

        return{
            "Intercept": intercept_,
            "Slope": slope_,
            "Predictions": test_predictions,
            "Training R-Squared": r_sq_train,
            "Test R-Squared": r_sq_test
        }
    

    
class LinearRegression:
    def __init__(self):
        self.beta_hat = None
        self.trained = False


    def train(self, train_data):
        '''
        Train a multiple linear regression model.

        :param train: Training dataset.

        :return: The beta hat and the r-squared for training set.
        '''
        predictions = np.array([])
        Y = train_data[:, -1]
        Z = train_data[:, :-1]
        Z = np.column_stack([np.ones(Z.shape[0]), Z])

        beta_hat = np.linalg.pinv(Z.T @ (Z)) @ (Z.T) @ (Y)
        self.beta_hat = beta_hat
        Y_hat = Z @ (beta_hat) 
        predictions = np.append(predictions, Y_hat)

        rss, tss = m_e.rss_tss(train_data, predictions)
        r_sq = m_e.r_squared(rss, tss)
        variance = m_e.var(beta_hat)

        self.trained = True

        return {
            "Beta_hat": self.beta_hat,
            "R-Squared Training": r_sq,
            "Variance": variance
        }
    

    def predict(self, test):
        """
        Test a multiple linear regression model.

        :param test: A test dataset.
        :param beta_hat: Beta hat value from the training model.

        :return: The prediction set and the r-squared for the test set.
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction.")

        predictions = np.array([])
        Z = test[:, :-1]
        Z = np.column_stack([np.ones(Z.shape[0]), Z])

        Y_hat = Z.dot(self.beta_hat) 
        predictions = np.append(predictions, Y_hat)

        rss, tss = m_e.rss_tss(test, predictions)
        r_sq = m_e.r_squared(rss, tss)
        variance = m_e.var(self.beta_hat)

        return{
            "Predictions": predictions,
            "R-Squared Test": r_sq,
            "Variance": variance
        }
    

