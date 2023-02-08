# Car Price Prediction using Linear Regression
This Python project consists of three programs that can be used to predict the price of a car based on its mileage.
The programs use linear regression to fit a line to a dataset of car prices and mileages,
and use the resulting line to make predictions.

## Linear Regression in non-technical terms
Linear regression is a statistical technique that helps us to understand the relationship between two or more variables. It involves finding a line that best fits a set of data points, and using that line to make predictions about future values.

Here is a simple example to illustrate the process of linear regression using mileage and price as the variables:

Imagine that you are trying to understand the relationship between the price of a car and its mileage. You might collect data on the price and mileage of a number of different cars, and plot this data on a graph. You might notice that, as the mileage of a car increases, its price tends to decrease.

To find the best-fit line for this data, we can use a mathematical formula that measures how far each data point is from the line. The goal is to find the line that minimizes the total distance between all of the data points and the line.

Once we have found the best-fit line, we can use it to make predictions about the price of a car based on its mileage. For example, if the best-fit line indicates that, for every 1,000 miles that a car has been driven, its price decreases by $500, we can use this information to predict the price of a car with a certain mileage.

Linear regression is a very useful tool for understanding and predicting the relationship between variables, and it is widely used in many different fields, including economics, finance, and biology.

## The Process of Linear Regression
The process of linear regression typically involves the following steps:

1. Collect and organize the data. This involves selecting the variables to be used in the model and collecting data points for those variables.

2. Choose a model. In linear regression, the model is a linear equation that describes the relationship between the dependent variable and the independent variables. The equation has the form `y = theta0 + theta1 * x1 + theta2 * x2 + ...`, where `y` is the dependent variable, `x1`, `x2`, etc. are the independent variables, and `theta0`, `theta1`, etc. are the parameters of the model (also known as the model coefficients).

3. Fit the model to the data. This involves finding the values of `theta0`, `theta1`, etc. that minimize the error between the predicted values of `y` and the actual values of `y`. This is typically done using an optimization algorithm, such as gradient descent.

4. Validate the model. Once the model has been fit to the data, it is important to validate it to ensure that it is accurate and reliable. This can be done by comparing the predicted values of `y` to the actual values of `y` and calculating the precision of the model.

5. Use the model to make predictions. After the model has been validated, it can be used to make predictions for new data points. Given a set of values for the independent variables, the model can be used to predict the corresponding value of the dependent variable.

## Requirements
To run these programs, you will need to have Python 3.9 and the following modules installed:

- matplotlib
- numpy

to install the libraries, run the following command:

```bash
pip3 install -r requirements.txt
```

## Program 1: predict.py
This program prompts the user for a mileage and returns the estimated price for that mileage based on the parameters in theta.csv. To use this program, run the following command:

```bash
python3 predict.py
```

## Program 2: train.py
This program reads the data from data.csv and performs linear regression on the data to find the best-fit line. It then saves the intercept and slope parameters of the line to theta.csv and plots the data and the line on a scatter plot. To use this program, run the following command:

```bash
python3 train.py
```