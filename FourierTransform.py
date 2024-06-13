# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:22:25 2024

@author: natha
"""

import cmath
import numpy as np

def chi(t, N):
    tt =  np.exp(2 * np.pi * complex(0, 1) * t / N)
    return(tt)

def single_freq_ft(f, m):
    N = len(f)
    ft = pow(N, -0.5) * sum([chi(-1 * m * x, N) * f[x] for x in range(N)])
    return(ft)

def fourier_transform(f):
    N = len(f)
    f_hat = [single_freq_ft(f, m) for m in range(N)]
    return(f_hat)

def single_freq_ift(f_hat, m):
    N = len(f_hat)
    ft = pow(N, -0.5) * sum([chi(m * x, N) * f_hat[x] for x in range(N)])
    return(ft)

def inverse_fourier_transform(f_hat):
    N = len(f_hat)
    f = [single_freq_ift(f_hat, m) for m in range(N)]
    return(f)

if __name__ == "__main__":
    print("Test Cases")    
    print('f -> f_hat -> f')
    for test in range(100):
        N = np.random.randint(2, 21)
        f = [complex(np.random.rand(), np.random.rand()) for m in range(N)]
        f_hat = fourier_transform(f)
        f2 = inverse_fourier_transform(f_hat)
        print(sum(np.abs(np.array(f) - np.array(f2)) > 0.001))

    print('f_hat -> f -> f_hat')
    for test in range(100):
        N = np.random.randint(2, 21)
        f_hat = [complex(np.random.rand(), np.random.rand()) for m in range(N)]
        f = inverse_fourier_transform(f_hat)
        f_hat2 = fourier_transform(f)
        print(sum(np.abs(np.array(f_hat) - np.array(f_hat2)) > 0.001))