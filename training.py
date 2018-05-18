#!/usr/bin/python
# coding: utf-8

__author__ = 'John Afaghpour'

"""
    Train using gradient descent algorithm. Normalize the dataset and adjusting learning rate
    over time speeds up training significantly. Bold driver algorithm
    is used to do so.
"""

from sys import argv
from dataset.action import *
import matplotlib.pyplot as plt


def hypothesis(x, theta):

    return float(theta[0] + (theta[1] * x))


def display(data, t0, t1, record):

    X = [min(data.km), max(data.km)]
    Y = []
    for m in X:
        m = t1 * normalize(m, min(data.km), max(data.km)) + t0
        price = denormalize(m, min(data.price), max(data.price))
        Y.append(price)

    fig = plt.figure(figsize=(25, 15), facecolor='beige')
    fig.canvas.set_window_title('Training records')
    plt.subplot(2, 2, 1)
    plt.plot(data.km, data.price, 'o')
    plt.plot(X, Y, linestyle='--', linewidth=4.0, color='green')
    plt.ylabel('Price')
    plt.xlabel('Mileage')
    plt.title('Linear regression')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(record['loss'], linewidth=3.0, color='green')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.title('Losses over time')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    theta0, = plt.plot(record['t0'], linewidth=3.0, label='t0', color='red')
    theta1, = plt.plot(record['t1'], linewidth=3.0, label='t1', color='blue')
    plt.legend([theta0, theta1], ['t0', 't1'], loc='best')
    plt.ylabel('Theta')
    plt.xlabel('Iteration')
    plt.title('Theta values over time')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(record['lr'], linestyle='-', linewidth=3.0, color='green')
    plt.ylabel('Learning rate')
    plt.xlabel('Iteration')
    plt.title('Learning rate over time')
    plt.grid(True)

    plt.show()


def bold_driver(LR, old, new):

    if old < new:
        return LR * 0.5, old
    return LR * 1.05, new


def get_cost(size, theta, km, price):

    total_error = 0
    for i in range(size):
        total_error += (price[i] - (theta[1] * km[i] + theta[0])) ** 2
    return total_error / float(size)


def gradient_descent(km, price, M, theta=[0.0, 0.0], i=90):

    record = {'t0':[], 't1':[], 'loss':[], 'lr':[]}
    size = int(M)
    LR = 1 / M * 2
    old = [0.0, 0.0]
    losses = [0.0, 0.0]
    for _ in range(i):
        old = [losses[0] / M, losses[1] / M]
        losses = [0.0, 0.0]
        for i in range(size):
            losses[0] += hypothesis(km[i], theta) - price[i]
            losses[1] += (hypothesis(km[i], theta) - price[i]) * km[i]
        losses[0] /= M
        losses[1] /= M
        LR, losses = bold_driver(LR, old, losses)
        theta[0] = theta[0] - LR * losses[0]
        theta[1] = theta[1] - LR * losses[1]
        record['loss'].append(get_cost(size, theta, km, price))
        record['t0'].append(theta[0])
        record['t1'].append(theta[1])
        record['lr'].append(LR)
    return theta, record


def main():

    graph = True if 'graph' in argv else False
    if not graph:
        print('pro tip: you can display graphs with "graph" argument')
    data = get_data('dataset/data.csv')
    km = normalize(data.km, min(data.km), max(data.km))
    price = normalize(data.price, min(data.price), max(data.price))
    try:
        theta, record = gradient_descent(km, price, float(len(data)))
        if graph is True:
            display(data, theta[0], theta[1], record)
    except (KeyboardInterrupt, SystemExit):
        exit('\nInterrupted')
    put_theta('dataset/theta.csv', theta)
    print("training done! t0: {} and t1: {} are stored in theta.csv".format(theta[0], theta[1]))

if __name__ == '__main__':
    main()
