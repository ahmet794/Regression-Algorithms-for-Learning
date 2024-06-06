#ridge_regression.py the ridge regression class and functions.
import numpy as np
import model_evaluation as m_e
from data_utils import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.beta_hat = None
        self.predictions = None
        self.r_squared = None
        self.variance = None
        self.mse = None

    def train(self, train_data):
        """
        Train and create a ridge regression model.

        :param train: A training dataset where the response variable is the last column.
        :param alpha: The tuning parameter.
        """
        Y = train_data[:, -1]
        Z = train_data[:, :-1]
        Z = np.column_stack([np.ones(Z.shape[0]), Z])

        I_p = np.eye(Z.shape[1])
        I_p[0, 0] = 0
        
        beta_hat = np.linalg.pinv(Z.T.dot(Z) + self.alpha * I_p).dot(Z.T).dot(Y)
        self.beta_hat = beta_hat
        self.predictions = Z.dot(beta_hat)

        rss, tss = m_e.rss_tss(train_data, self.predictions)
        self.r_squared = m_e.r_squared(rss, tss)
        self.variance = m_e.var(beta_hat)
        self.mse = m_e.mse(Y, self.predictions)

    
    def coef_(self):
        """
        The coefficients of the trained model.

        :return: A list of coefficient values of the model.
        """
        return self.beta_hat[1:]
    
    def intercept(self):
        """
        The intercept of the trained model.

        :return: A float, the intercept of the model.
        """
        return self.beta_hat[0]
    
    def r_score(self):
        """
        The R-Squared score of the model.

        :return: A float, the R-Squared score of the model.
        """
        return self.r_squared
    
    def var(self):
        """
        The variance of the model.

        :return: A float, the variance of the model.
        """
        return self.variance
    
    def predict(self, data):
        """
        The predictions that are made based on the model.

        :param data: The data that predictions should be made on.

        :return: A list of values for the predictions.
        """
        if self.beta_hat is None:
            raise ValueError("Model must be trained before prediction.")
        Z = data[:, :-1]
        Z = np.column_stack([np.ones(data.shape[0]), Z])
        return Z.dot(self.beta_hat)
    
    def predictGUI(self, data):
        """
        The prediction that are made for the GUI.

        :param data: The input data that is given from the user.

        :return: The prediction that are made by the model that was trained.
        """
        if self.beta_hat is None:
            raise ValueError("Model must be trained before prediction.")
        Z = np.column_stack([np.ones(data.shape[0]), data])
        return Z.dot(self.beta_hat)
    
    def mse_(self):
        """
        The mean squared error of the model.

        :return: The mean squared error.
        """
        return self.mse


    def test(self, test_data):
        """
        Testing an existing ridge model.

        :param test: A test set that has the last columns as the response variable.
        :param beta_hat: A list of coefficients from the trained ridge model.
        """
        if self.beta_hat is None:
            raise ValueError("Model must be trianed before prediction.")
        
        Y = test_data[:, -1]
        predictions = np.array([])
        Z = test_data[:, :-1]
        Z = np.column_stack([np.ones(Z.shape[0]), Z])
        
        Y_hat = Z.dot(self.beta_hat)
        predictions = np.append(predictions, Y_hat)

        rss, tss = m_e.rss_tss(test_data, predictions)
        self.r_squared = m_e.r_squared(rss, tss)
        self.variance = m_e.var(self.beta_hat)
        self.mse = m_e.mse(Y, Y_hat)
    
    def nonconformity_scores(self, data, predictions):
        """
        Measuring the nonconformity scores of the model for conformal prediction.

        :param data: The training data that is applied to the model.

        :return: A sorted list of nonconformity scores.
        """
        if self.beta_hat is None:
            raise ValueError("Model must be trianed before measuring nonconformity scores.")
        y = data[:, -1]
        y_hat = predictions

        return sorted(np.abs(y_hat - y))
    
    def p_vals(self, nonconformity_scores, train_data, new_data, predictions):
        """
        Estimate the p-values of the new house prices.

        :param nonconformity_scores: A list of nonconformity scores from the training dataset.
        :param train_data: The training data that was used to create the model.
        :param new_data: New data that is introduced to make predictions about.
        :param predictions: The predictions that the model makes for the new data that is introduced.
    
        :return: A list of p-values of the predictions that are made.
        """
        p_values = []
        n = train_data.shape[0]
        y = new_data[:, -1]
        test_nonconformity_scores = np.abs(y - predictions)
        for test_nonconformity_score in test_nonconformity_scores:
            count = 0
            for nonconformity_score in nonconformity_scores:
                if test_nonconformity_score >= nonconformity_score:
                    count += 1
            p_value = count/n
            p_values.append(p_value)
        return p_values

    def conformal_prediction(self, train_data, calibration_data, test_data, confidence_level):
        """
        Generate conformal prediction intervals for new data.
        
        :param train_data: Training data to train the model.
        :param calibration_data: A data that is not the same as the training data, to calibrate the conformal prediction intervals.
        :param test_data: A data to make predictions and create confidence intervals.
        :param confidence_level: The confidence level for the prediction intervals.

        :return: A list of tuples with lower and upper bounds of the prediction.
        """

        self.train(train_data)

        # Using the calibration data, estimate the nonconformity scores
        calibration_predictions = self.predict(calibration_data)
        nonconformity_scores = self.nonconformity_scores(calibration_data, calibration_predictions)

        # Apply the test data to make predictions
        test_predictions = self.predict(test_data)

        # Create intervals using the nonconformity scores
        prediction_intervals = []
        quantile = np.quantile(nonconformity_scores, confidence_level) # Given the confidence level, returns specific quantile according to the nonconformity scores
        for prediction in test_predictions:
            lower_bound = prediction - quantile
            upper_bound = prediction + quantile
            prediction_intervals.append((lower_bound, upper_bound))
        return prediction_intervals
    
    def conformal_predictionGUI(self, prediction, calibration_data, confidence_level):
        """
        Generate conformal prediction intervals for new data.
        
        :param prediction: The prediction value obtained by the model.
        :param calibration_data: A data that is not the same as the training data, to calibrate the conformal prediction intervals.
        :param confidence_level: The confidence level for the prediction intervals.

        :return: A list of tuples with lower and upper bounds of the prediction.
        """
        # Using the calibration data, estimate the nonconformity scores
        calibration_predictions = self.predict(calibration_data)
        nonconformity_scores = self.nonconformity_scores(calibration_data, calibration_predictions)

        # Create intervals using the nonconformity scores
        prediction_intervals = []
        quantile = np.quantile(nonconformity_scores, confidence_level) # Given the confidence level, returns specific quantile according to the nonconformity scores

        # Create bounds for the interval
        lower_bound = prediction - quantile
        upper_bound = prediction + quantile
        prediction_intervals.append((lower_bound, upper_bound))
        return prediction_intervals


    def cross_conformal_prediction(self, k, data, new_data, confidence_level):        
        """
        Generate conformal prediction intervals for the new data by applying cross-validation.

        :param k: The fold size for cross-validation.
        :param data: The dataset for the model to be trained, calibrated and tested on.
        :param new_data: The new data that is introduced for predictions.
        :param confidence_level: The confidence level for the conformal prediction intervals.
        
        :return: A list of tuples with lower and upper bounds of the prediction.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=17)
        scaler = StandardScaler()

        # This will accumulate all nonconformity scores across all folds
        nonconformity_scores = []

        # Perform k-fold cross-validation
        for train_index, test_index in kf.split(data):
            # Split data to training and calibration data
            train_data, calibration_data = data[train_index], data[test_index]

            # Scale data
            X_train, Y_train = scaler.fit_transform(train_data[:, :-1]), train_data[:, -1] 
            X_cal, Y_cal = scaler.transform(calibration_data[:, :-1]), calibration_data[:, -1]

            # Recombine feature and target data
            train_data = np.hstack((X_train, Y_train.reshape(-1, 1)))
            calibration_data = np.hstack((X_cal, Y_cal.reshape(-1, 1)))
           
            # Train data and estimate the nonconformity scores based on calibration data
            self.train(train_data)
            calibration_predictions = self.predict(calibration_data)
            nc = self.nonconformity_scores(calibration_data, calibration_predictions)
            nonconformity_scores.extend(nc)
        
        # Train the model on the whole dataset and predict values on a given new data
        self.train(data)
        test_predictions = self.predict(new_data)

        # Similar to conformal prediction function the intervals are estimated using quantiles
        prediction_intervals = []
        quantile = np.quantile(nonconformity_scores, confidence_level)
        for prediction in test_predictions:
            lower_bound = prediction - quantile
            upper_bound = prediction + quantile
            prediction_intervals.append((lower_bound, upper_bound))
        return prediction_intervals
    
    def calculate_error_rate(self, test_data, prediction_intervals):
        """
        Calculate the error rates given prediction intervals and the data that is introduced to the model.

        :param test_data: The data that the model predicted on.
        :parma prediction_intervals: The prediction intervals that acquired by conformal prediction.

        :return: Return the error rate that should be a close value to confidence level.
        """
        y = test_data[:, -1]
        errors = [y_value < lower_bound or y_value > upper_bound for y_value, (lower_bound, upper_bound) in zip(y, prediction_intervals)]
        error_rate = np.mean(errors)
        return error_rate
    
    def conformal_prediction_calibration_curve(model, train_data, calibration_data, test_data, confidence_levels):
        """
        Plots a calibration curve for conformal prediction intervals.

        Parameters:
        :param model: An instance of the learning algorithms that has the method 'conformal_prediction'.
        :param train_data: The dataset used to generate conformal prediction intervals via cross-validation.
        :param calibration_data: New data points for which the conformal prediction intervals have been calculated.
        :param test_data: The data for testing the algorithm.
        :param confidence_levels: An array of confidence levels to evaluate.
        """
        observed_confidence_levels = []

        for alpha in confidence_levels:
            # Generate prediction intervals for the current confidence level
            prediction_intervals = model.conformal_prediction(train_data, calibration_data, test_data, confidence_level=alpha)
            
            # Count how many of the actual new data values fall within their corresponding prediction intervals
            actuals = test_data[:, -1]
            within_interval_count = sum([1 for i, interval in enumerate(prediction_intervals) if interval[0] <= actuals[i] <= interval[1]])
            
            # Calculate the observed confidence level
            observed_confidence_level = within_interval_count / len(test_data)
            observed_confidence_levels.append(observed_confidence_level)
        
        # Plot the calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(confidence_levels, observed_confidence_levels, color='blue', label='Observed Confidence Level')
        plt.plot(confidence_levels, confidence_levels, linestyle='--', color='red', label='Expected Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Proportion of Actuals Within Prediction Interval')
        plt.title('Calibration Curve for Conformal Prediction Intervals')
        plt.legend()
        plt.show()


    def cross_conformal_calibration_curve(model, data, new_data, confidence_levels):
        """
        Plots a calibration curve for conformal prediction intervals.

        :param model: An instance of the RidgeRegression class that has the method 'cross_conformal_prediction'.
        :param data: The dataset used to generate conformal prediction intervals via cross-validation.
        :param new_data: New data points for which the conformal prediction intervals have been calculated.
        :param confidence_levels: An array of confidence levels to evaluate.
        """
        observed_confidence_levels = []

        for alpha in confidence_levels:
            # Generate prediction intervals for the current confidence level
            prediction_intervals = model.cross_conformal_prediction(k=5, data=data, new_data=new_data, confidence_level=alpha)
            
            # Count how many of the actual new data values fall within their corresponding prediction intervals
            actuals = new_data[:, -1]
            within_interval_count = sum([1 for i, interval in enumerate(prediction_intervals) if interval[0] <= actuals[i] <= interval[1]])
            
            # Calculate the observed confidence level
            observed_confidence_level = within_interval_count / len(new_data)
            observed_confidence_levels.append(observed_confidence_level)
        
        # Plot the calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(confidence_levels, observed_confidence_levels, color='blue', label='Observed Confidence Level')
        plt.plot(confidence_levels, confidence_levels, linestyle='--', color='red', label='Expected Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Proportion of Actuals Within Prediction Interval')
        plt.title('Calibration Curve for Conformal Prediction Intervals')
        plt.legend()
        plt.show()
            







