######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.shape

df.head()

X = df[["TV"]]
y = df[["sales"]]


##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV

# intercept (b - bias)
reg_model.intercept_[0]
# 7.03

# coefficient of tv (w1)
reg_model.coef_[0][0]
#0.04

##########################
# Prediction
##########################

# How many sales would be expected if TV spending was 150 units?
reg_model.intercept_[0] + reg_model.coef_[0][0]*150
# 14.16

# How many sales would there be if there was a TV spending of 500 units?
reg_model.intercept_[0] + reg_model.coef_[0][0]*500
# 30.80

df.describe().T


# Visualization of the Model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV Spending")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


##########################
# Prediction Success
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51

y.mean()
# 14.02

y.std()
# 5.22

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-Square
reg_model.score(X, y)
# 0.61

######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1)

y = df[["sales"]]


##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape # (160, 3)
X_test.shape # (40, 3)

y_train.shape # (160, 1)
y_test.shape # (40, 1)

reg_model = LinearRegression().fit(X_train, y_train)

# intercept (b - bias)
reg_model.intercept_
# 2.90794702

# coefficients (w - weights)
reg_model.coef_
# [0.0468431 , 0.17854434, 0.00258619]

##########################
# Prediction
##########################

# What is the expected value of the sale based on the following observation values?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90794702
# 0.0468431 , 0.17854434, 0.00258619

# Sales = bias + ( TV * w1 (tv) + radio * w2 (radio) + newspaper * w3 (newspaper) )
# Sales = 2.90794702 + ( TV * 0.0468431 + radio * 0.17854434 + newspaper * 0.00258619 )

2.90794702 + ( 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 )
# 6.20213102

new_data = [[30], [10], [40]]
# [[30], [10], [40]]

new_data = pd.DataFrame(new_data).T
#     0   1   2
# 0  30  10  40

reg_model.predict(new_data)
# [6.202131]

##########################
# Evaluating Prediction Success
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN R square
reg_model.score(X_train, y_train)
# 0.89

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test Rsquare
reg_model.score(X_test, y_test)
# 0.89

# 10x CV(Determines the cross-validation splitting strategy) RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69

# 5x CV(Determines the cross-validation splitting strategy) RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71

######################################################
# Simple Linear Regression with Gradient Descent from Scratch ( Just Example )
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    '''
    Calculates the Mean Squared Error (MSE) for a simple linear regression model.

    Parameters:
        Y (list or array-like): Actual target values.
        b (float): Bias term (intercept) of the model.
        w (float): Weight term (slope) of the model.
        X (list or array-like): Input feature values.

    Returns:
        float: The mean squared error between predicted and actual values.

    This function computes the predictions using the linear model y_hat = b + w * X[i],
    then calculates the squared differences from the actual values Y[i], and returns
    the average of these squared errors.
    '''
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    '''
    Performs one step of gradient descent to update the weights of a simple linear regression model.

    Parameters:
        Y (list or array-like): Actual target values.
        b (float): Current bias term (intercept).
        w (float): Current weight term (slope).
        X (list or array-like): Input feature values.
        learning_rate (float): Learning rate for gradient descent.

    Returns:
        tuple: A tuple (new_b, new_w) containing the updated bias and weight values.

    This function computes the gradients of the cost function with respect to the bias and weight,
    then updates them using the gradient descent algorithm to minimize the mean squared error.
    '''
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    '''
    Trains a simple linear regression model using gradient descent.

    Parameters:
        Y (list or array-like): Actual target values.
        initial_b (float): Initial value for the bias term (intercept).
        initial_w (float): Initial value for the weight term (slope).
        X (list or array-like): Input feature values.
        learning_rate (float): Learning rate for gradient descent updates.
        num_iters (int): Number of iterations to run gradient descent.

    Returns:
        tuple: A tuple (cost_history, b, w) where:
            - cost_history is a list of MSE values at each iteration,
            - b is the final bias value,
            - w is the final weight value.

    This function performs iterative updates of the model parameters using the gradient descent algorithm.
    It prints the progress every 100 iterations and at the beginning and end of training.
    '''

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.01
initial_w = 0.01
num_iters = 50000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

# learning_rate = 0.001
# initial_b = 0.001
# initial_w = 0.001

# After 10000 iterations b = 8.79062282139543, w = 0.2184390657183005, mse = 18.170680975262957
# After 50000 iterations b = 9.311632953405377, w = 0.20249594073219987, mse = 18.092397745133063

# learning_rate = 0.01
# initial_b = 0.01
# initial_w = 0.01

# After 10000 iterations b = 8.791111365807714, w = 0.21842411605773202, mse = 18.17053423521168
# After 50000 iterations b = 9.311632958226651, w = 0.2024959405846669, mse = 18.09239774513305