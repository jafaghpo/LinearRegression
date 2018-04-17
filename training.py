#!/usr/bin/python3
# coding: utf-8

__version__ = '1.0'
__author__ = 'John Afaghpour'

"""
    Train using gradient descent algorithm. Normalize the dataset and ajusting learning rate
    over time speeds up training significantly. Bold driver algorithm
    is used to do so.
"""

import dataset
from sys import argv
import matplotlib.pyplot as plt


def hypothesis(x, theta):

    return float(theta[0] + (theta[1] * x))


def display(data, t0, t1, history):

    X = [min(data.km), max(data.km)]
    Y = []
    for m in X:
        m = t1 * dataset.normalize(m, min(data.km), max(data.km)) + t0
        price = dataset.denormalize(m, min(data.price), max(data.price))
        Y.append(price)

    plt.figure(1)
    plt.plot(data.km, data.price, 'o')
    plt.plot(X, Y, linestyle='--', linewidth=4.0)
    plt.ylabel('Price')
    plt.xlabel('Mileage')

    plt.figure(2)
    plt.plot(history['loss'], linestyle=':', linewidth=4.0)
    plt.ylabel('loss')
    plt.xlabel('iteration')

    plt.figure(3)
    plt.plot(history['t0'], linestyle=':', linewidth=4.0)
    plt.ylabel('t0')
    plt.xlabel('iteration')

    plt.figure(4)
    plt.plot(history['t1'], linestyle=':', linewidth=4.0)
    plt.ylabel('t1')
    plt.xlabel('iteration')
    plt.show()


def bold_driver(lr, old, new):

    if old < new:
        return lr * 0.5, old
    return lr * 1.05, new


def cost_func(m, theta, km, price):

    total_error = 0
    for i in range(m):
        total_error += (price[i] - (theta[1] * km[i] + theta[0])) ** 2
    return total_error / float(m)


def gradient_descent(km, price, m, theta=[0.0, 0.0], i=90):

    history = {'t0':[], 't1':[], 'loss':[]}
    lr = 1 / m * 2
    old = [0.0, 0.0]
    losses = [0.0, 0.0]
    for _ in range(i):
        old = [losses[0] / float(m), losses[1] / float(m)]
        losses = [0.0, 0.0]
        for i in range(m):
            losses[0] += hypothesis(km[i], theta) - price[i]
            losses[1] += (hypothesis(km[i], theta) - price[i]) * km[i]
        losses[0] /= float(m)
        losses[1] /= float(m)
        lr, losses = bold_driver(lr, old, losses)
        theta[0] = theta[0] - lr * losses[0]
        theta[1] = theta[1] - lr * losses[1]
        history['loss'].append(cost_func(m, theta, km, price))
        history['t0'].append(theta[0])
        history['t1'].append(theta[1])
    return theta, history

graph = True if 'graph' in argv else False
if not graph:
    print('pro tip: you can display graphs with "graph" argument')

if __name__ == '__main__':
    data = dataset.get_training('data.csv')
    km = dataset.normalize(data.km, min(data.km), max(data.km))
    price = dataset.normalize(data.price, min(data.price), max(data.price))
    try:
        theta, history = gradient_descent(km, price, len(data))
        if graph is True:
            display(data, theta[0], theta[1], history)
    except (KeyboardInterrupt, SystemExit):
        exit('\nInterrupted')
    dataset.put_theta('theta.csv', theta)
