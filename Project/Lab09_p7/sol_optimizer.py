# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:59:28 2024

@author: anjal
"""

import numpy
import scipy.optimize

def f(x):
    y,z = x
    return (y+3)**2 + numpy.sin(y) + (z+1)**2

def fprime(x):
    y,z = x
    return numpy.array([2*(y+3) + numpy.cos(y), 2 * (z+1)])

print (scipy.optimize.fmin_l_bfgs_b(func = f, approx_grad = True, x0 = numpy.zeros(2)))
print (scipy.optimize.fmin_l_bfgs_b(func = f, fprime = fprime, x0 = numpy.zeros(2)))