#!/usr/bin/env python3
# coding: utf-8

__version__ = '0.1'
__author__ = 'John Afaghpour'

"""
    Estimates the price of a car based on mileage,
    using thetas value (weight for training) stored
    in theta.csv. Values are previously normalized to
    optimize training speed.
"""

import dataset


def estimate_price(data, theta):

    try:
        mileage = input("Please enter a mileage to estimate the price: ")
        mileage = float(mileage)
    except (TypeError, NameError) as e:
        exit('error: mileage is not valid')
    except (KeyboardInterrupt, SystemExit):
        exit('\nInterrupted')
    if mileage < 0:
        print('A mileage cannot be negative, try again')
        return estimate_price(data, theta)
    try:
        mileage = dataset.normalize(mileage, min(data.km), max(data.km))
    except ZeroDivisionError as e:
        exit('error: {}'.format(e))
    price = theta[0] + theta[1] * mileage
    if price > 0:
        price = dataset.denormalize(price, min(data.price), max(data.price))
    return price if price > 0 else 0


if __name__ == '__main__':
    data = dataset.get_training('data.csv')
    theta = dataset.get_theta('theta.csv')
    price = estimate_price(data, theta)
    print('Estimated price is {}$'.format(price))
