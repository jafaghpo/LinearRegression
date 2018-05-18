#!/usr/bin/python
# coding: utf-8

__author__ = 'John Afaghpour'

"""
    Estimates the price of a car based on mileage,
    using thetas value (weight for training) stored
    in theta.csv. Values are previously normalized to
    optimize training speed.
"""

from dataset.action import *


def estimate_price(data, theta):

    try:
        mileage = input("Please enter a mileage to estimate the car price: ")
        mileage = float(mileage)
    except (TypeError, NameError, SyntaxError) as e:
        exit('error: mileage is not valid')
    except (KeyboardInterrupt, SystemExit):
        exit('\nInterrupted')
    if mileage < 0:
        print('A mileage cannot be negative, try again')
        return estimate_price(data, theta)
    try:
        mileage = normalize(mileage, min(data.km), max(data.km))
    except ZeroDivisionError as e:
        exit('error: {}'.format(e))
    price = theta[0] + theta[1] * mileage
    if price > 0:
        price = denormalize(price, min(data.price), max(data.price))
    return price if price > 0 else 0


def main():

    data = get_data('dataset/data.csv')
    theta = get_theta('dataset/theta.csv')
    price = estimate_price(data, theta)
    print('This car is worth %d$' % price)

if __name__ == '__main__':
    main()
