# Final Year Project Repository

## Topic: Regression Algorithms for Learning

### Ahmet Cihan

## Overview

This repository contains implementations and comparisons of various regression algorithms, focusing on machine learning techniques to handle multicollinearity and feature selection within different housing datasets.

## Algorithms Implemented

- Linear Regression
- Ridge Regression
- Lasso Regression
- Gradient Descent Method
- Conformal Prediction
- Cross-Conformal Prediction

## Datasets

Algorithms are applied to datasets such as 
- New York Housing, 
- Washington Housing, 
- Boston Housing, 
- Extended Boston Housing, 
- California Housing, 
representing diverse regression challenges.

## Requirements

- scikit-learn
- matplotlib
- PySide6
- numpy
- pandas

When run in a new environment, executing app.py will automatically download any missing or required libraries for the project.

## Installation

1. Clone the repository:

git clone https://gitlab.cim.rhul.ac.uk/zkac149/PROJECT.git

2. Run app.py

python3 app.py

## Structure

The project is organized into distinct Python modules, each with specific functionalities:

- `__init__.py`: Initializes the directory as a Python package, allowing for module imports across the project.
- `app.py`: Acts as the entry point of the application, checking and installing any missing dependencies upon execution.
- `data_utils.py`: Provides utility functions for data operations such as loading, preprocessing, normalization, and feature engineering.
- `gradient_descent.py`: Implements the gradient descent optimization algorithm for regression model fitting.
- `lasso_regression.py`: Implements Lasso Regression, employing a penalty term to reduce certain coefficients to zero, aiding in feature selection.
- `linear_r.py`: Contains the fundamental setup for performing Linear Regression from the ground up.
- `linear_regression.py`: The core module for linear regression analysis, featuring model fitting and predictive capabilities.
- `model_evaluation.py`: Offers a suite of functions for evaluating model performance with metrics such as R² and MSE.
- `model_selection.py`: Equipped with tools for model selection and hyperparameter tuning to improve model performance.
- `plotting.py`: A utility module for generating visual representations of data, regression models, and their performance.
- `requirements.txt`: Enumerates all dependencies required to run the project, facilitating environment setup.
- `ridge_regression.py`: Provides an implementation of Ridge Regression, addressing multicollinearity with L2 regularization.

- Jupyter notebooks - Detailed explanations, testing and visualizations of the models.

## Conformal Prediction

Emphasizes conformal prediction techniques to provide reliable predictions with confidence intervals.

## Software Engineering Practices

Adheres to best practices, including modular design, version control, and comprehensive testing.

