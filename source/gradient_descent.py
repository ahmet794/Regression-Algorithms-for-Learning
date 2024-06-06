# gradient_descent.py has the optimization algorithm for minimising the cost function.

import model_evaluation
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(data, learning_rate, threshold, mse_test):
    """
    Computes the parameters of a given data using mse as a cost function.
    """
    Z = data[:, :-1] 
    ones = np.ones((Z.shape[0], 1))
    Z = np.hstack((ones, Z))
    Y = data[:, -1]

    theta = np.zeros(Z.shape[1]) 
    cost_function = []

    iteration = 0
    while True:
        y_hat = Z@(theta)
        mse = model_evaluation.mse(Y, y_hat)
        cost_function.append(mse)

                # Debugging Information
        if iteration % mse_test== 0:
            print(f"Iteration: {iteration}, MSE: {mse}")

        gradient = 2/len(Z)*Z.T@(y_hat - Y)
        gradient_norm = np.linalg.norm(gradient)
        theta -= learning_rate*gradient

        if gradient_norm < threshold:
            break
        iteration += 1

    rss, tss = model_evaluation.rss_tss(data, y_hat)
    plt.figure()
    plt.plot(cost_function, label='MSE')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.legend()
    plt.show()

    return theta, 'R-Squared Score:', model_evaluation.r_squared(rss, tss)
                   

def stochastic_g_d(data, learning_rate, epochs, threshold):
    """
    Using the stochastic gradient descent method, computes the parameters of a given data using mse as a cost function.
    """
    Z = data[:, :-1] 
    ones = np.ones((Z.shape[0], 1))
    Z = np.hstack((ones, Z))
    Y = data[:, -1]

    theta = np.zeros(Z.shape[1]) 
    cost_function = []

    for epoch in range(epochs):
        cost = 0
        gradients = []
        for i in range(len(Z)):
            random_index = np.random.randint(len(Z))
            xi = Z[random_index:random_index+1]
            yi = Y[random_index:random_index+1]

            y_hat = xi@(theta)
            gradient = 2*xi.T@(y_hat - yi)
            gradients.append(gradient)
            theta -= learning_rate*gradient

            cost += np.mean((yi - y_hat)**2)
        
        avg_gradient = np.mean(gradients, axis=0)
        gradient_norm = np.linalg.norm(avg_gradient)
        cost_function.append(cost/len(Z))

        if gradient_norm < threshold:
            break

    return theta

def lasso_gradient_descent(data, learning_rate, threshold, mse_test, lasso_scalar, max_iterations = 1000000):
    """
    Computes the parameters of a given data using mse as a cost function.
    """
    Z = data[:, :-1] 
    ones = np.ones((Z.shape[0], 1))
    Z = np.hstack((ones, Z))
    Y = data[:, -1]

    theta = np.zeros(Z.shape[1]) 
    cost_function = []

    iteration = 0

    for i in range(max_iterations):
        y_hat = Z@(theta)
        mse = model_evaluation.mse(Y, y_hat)
        if np.isnan(mse):
            return 'divergence'
        cost_function.append(mse)

        if iteration % mse_test== 0:
            print(f"Iteration: {iteration}, MSE: {mse}")

        gradient = 2/len(Z)*Z.T@(y_hat - Y) + lasso_scalar*np.sign(theta)
        gradient_norm = np.linalg.norm(gradient)
        theta -= learning_rate*gradient

        if gradient_norm < threshold:
            break

        iteration += 1

    rss, tss = model_evaluation.rss_tss(data, y_hat)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(cost_function, label='MSE')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(len(theta)), theta, color='skyblue')
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title(f'Lasso Regression Coeffcients, $\lambda$ = {lasso_scalar}')
    plt.tight_layout()

    plt.show()


    return theta, model_evaluation.r_squared(rss, tss)

