#!/usr/bin/python
# coding: utf-8

__author__ = 'John Afaghpour'

"""
    Tools functions to manage datasets.
"""

from pandas import read_csv


def normalize(x, min, max):

    return (x - min) / (max - min)


def denormalize(x, min, max):

    return x * (max - min) + min


def get_data(path):

    try:
        data = read_csv(path, dtype='float')
    except Exception as e:
        exit('error: {}'.format(e))
    return data


def get_theta(path):

    try:
        data = read_csv(path, dtype='float')
        theta = [float(data.t0), float(data.t1)]
    except Exception:
        print('theta.csv is missing, you should train before estimating a price.')
        print('Setting theta values to zero ...')
        theta = [0., 0.]
    return theta


def put_theta(path, theta):

    with open(path, 'w') as f:
        f.write('t0,t1\n{},{}\n'.format(theta[0], theta[1]))

if __name__ == '__main__':
    print('This is a module containing only functions so there is no point in running this file.')

