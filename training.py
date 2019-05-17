#!python3
# coding: utf-8

__author__ = 'John Afaghpour'

"""
    Train using gradient descent algorithm. Normalize the dataset and adjusting learning rate
    over time speeds up training significantly. Bold driver algorithm
    is used to do so.
"""

from sys import argv
import parser
import matplotlib.pyplot as plt

def hypothesis(x, theta):

    return float(theta[0] + (theta[1] * x))


def display(data, t0, t1, record):

    X = [min(data.km), max(data.km)]
    Y = []
    for m in X:
        m = t1 * parser.normalize(m, min(data.km), max(data.km)) + t0
        price = parser.denormalize(m, min(data.price), max(data.price))
        Y.append(price)

    fig = plt.figure(figsize=(23, 12))
    fig.canvas.set_window_title('Training records')
    plt.subplot(2, 2, 1)
    plt.plot(data.km, data.price, 'o')
    plt.plot(X, Y, linestyle='--', linewidth=4.0, color='green')
    plt.ylabel('Price')
    plt.xlabel('Mileage')
    plt.title('Best fit line')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(record['error'], linewidth=3.0, color='green')
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.title('Error over time')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    theta0, = plt.plot(record['t0'], linewidth=3.0, label='t0', color='red')
    theta1, = plt.plot(record['t1'], linewidth=3.0, label='t1', color='blue')
    plt.legend([theta0, theta1], ['t0', 't1'], loc='best')
    plt.ylabel('Theta')
    plt.xlabel('Iterations')
    plt.title('Theta values over time')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(record['lr'], linestyle='-', linewidth=3.0, color='green')
    plt.ylabel('Learning rate')
    plt.xlabel('Iterations')
    plt.title('Learning rate over time')
    plt.grid(True)

    plt.show()


def bold_driver(LR, old, new):

    if old < new:
        return LR * 0.5, old
    return LR * 1.05, new


def get_error(size, theta, km, price):

    total_error = 0
    for i in range(size):
        total_error += (price[i] - (theta[1] * km[i] + theta[0])) ** 2
    return total_error / float(size)


def gradient_descent(km, price, M, theta=[0.0, 0.0]):

    record = {'t0':[], 't1':[], 'error':[], 'lr':[]}
    size = int(M)
    LR = (1 / M * 2)
    old = [0.0, 0.0]
    sum_theta = [0.0, 0.0]
    prev_error = 0.0
    while True:
        old = [sum_theta[0] / M, sum_theta[1] / M]
        sum_theta = [0.0, 0.0]
        for i in range(size):
            sum_theta[0] += hypothesis(km[i], theta) - price[i]
            sum_theta[1] += (hypothesis(km[i], theta) - price[i]) * km[i]
        sum_theta[0] /= M
        sum_theta[1] /= M
        LR, sum_theta = bold_driver(LR, old, sum_theta)
        theta[0] = theta[0] - LR * sum_theta[0]
        theta[1] = theta[1] - LR * sum_theta[1]
        error = get_error(size, theta, km, price)
        record['error'].append(error)
        record['t0'].append(theta[0])
        record['t1'].append(theta[1])
        record['lr'].append(LR)
        if abs(error - prev_error) < 0.000001:
            return theta, record
        prev_error = error


def main():

    graph = True if 'graph' in argv else False
    if not graph:
        print('pro tip: you can display graphs with "graph" argument')
    data = parser.get_data('data.csv')
    km = parser.normalize(data.km, min(data.km), max(data.km))
    price = parser.normalize(data.price, min(data.price), max(data.price))
    try:
        theta, record = gradient_descent(km, price, float(len(data)))
        if graph is True:
            display(data, theta[0], theta[1], record)
    except (KeyboardInterrupt, SystemExit):
        exit('\nInterrupted')
    parser.store_theta('theta.csv', theta)
    print("training done! t0: {} and t1: {} are stored in theta.csv".format(theta[0], theta[1]))

if __name__ == '__main__':
    main()
