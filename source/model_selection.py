# model_selection.py is dedicated to have functionalities to pick optimal models.
import numpy as np
import model_evaluation as m_e
from ridge_regression import RidgeRegression
from data_utils import StandardScaler


def k_fold_cross_validation(model, data, k, score="R-Squared"):
    """
    Using a number of value to divide the data to various training, and test data.

    :param model: A learning model.
    :param data: A complete data without being divided into trianing and testing.
    :param k: An integer to divide the data how many times for k-fold cross validation.

    :return: The mean of the Mean Squared Errors depending on the training and test sets.
    """
    fold_size = len(data) // k # rounding to the nearest integer
    train_metrics = []
    test_metrics = []
    scaler = StandardScaler()

    for i in range(k):
        start, end = i * fold_size, (i+1) * fold_size
        test_data = data[start:end]
        train_data = np.concatenate([data[:start], data[end:]])

        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)

        if score == "R-Squared":
            model.train(train_data_scaled)
            train_score = model.r_score()
            train_metrics.append(train_score)
            model.test(test_data_scaled)
            test_score = model.r_score()
            test_metrics.append(test_score)
        
        elif score == "MSE":
            model.train(train_data_scaled)
            train_score = model.mse
            train_metrics.append(train_score)
            model.test(test_data_scaled)
            test_score = model.mse
            test_metrics.append(test_score)
            
    return np.mean(train_metrics), np.mean(test_metrics)


def lambda_ridge(data, k, lambda_values, score = "MSE"):
    best_lambda = None
    best_score = None
    if score=='MSE':
        for l in lambda_values:
            model = RidgeRegression(alpha=l)
            train_score, cv_score = k_fold_cross_validation(model, data, k, score=score)
            print(cv_score)

            if best_score is None or cv_score < best_score:
                best_lambda = l
                best_score = cv_score
                best_train = train_score
    elif score=='R-Squared':
        for l in lambda_values:
            model = RidgeRegression(alpha=l)
            train_score, cv_score = k_fold_cross_validation(model, data, k, score=score)
            print(cv_score)

            if best_score is None or cv_score > best_score:
                best_lambda = l
                best_score = cv_score
                best_train = train_score
            
    print(f'Best lambda: {best_lambda}, with score: {best_score}' )
    print("Training Score: ", best_train, " Test Score: ", best_score)
    final_model = RidgeRegression(alpha=best_lambda)
    return final_model

def lambda_lasso(data, k, lambda_values, lasso_model, score='MSE'):
    best_lambda = None
    best_score = None

    if score == 'MSE':
        for l in lambda_values:
            lasso_model.lasso_lambda = l
            train_score, cv_score = k_fold_cross_validation(lasso_model, data, k, score=score)
            print(cv_score)

            if best_score is None or cv_score < best_score:
                best_lambda = l
                best_score = cv_score
                best_train = cv_score
    elif score == 'R-Squared':
        for l in lambda_values:
            lasso_model.lasso_lambda = l
            train_score, cv_score = k_fold_cross_validation(lasso_model, data, k, score=score)
            print(cv_score)

            if best_score is None or cv_score > best_score:
                best_lambda = l
                best_score = cv_score
                best_train = train_score

    print(f'Best lambda: {best_lambda}, with score: {best_score}')
    print("Training Score: ", best_train, " Test Score: ", best_score)
    lasso_model.lasso_lambda = best_lambda
    final_model = lasso_model
    return final_model





